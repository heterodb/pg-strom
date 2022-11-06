/*
 * dpuserv.c
 *
 * A standalone command that handles XpuCommands on DPU devices
 * --------
 * Copyright 2011-2021 (C) KaiGai Kohei <kaigai@kaigai.gr.jp>
 * Copyright 2014-2021 (C) PG-Strom Developers Team
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the PostgreSQL License.
 */
#include "dpuserv.h"

#define PEER_ADDR_LEN	80
typedef struct
{
	dlist_node			chain;	/* link to dpu_client_list */
	kern_session_info  *session;/* per-session information */
	volatile int32_t	refcnt;	/* odd-number as long as socket is active */
	pthread_mutex_t		mutex;	/* mutex to write the socket */
	int					sockfd;	/* connection to PG-backend */
	pthread_t			worker;	/* receiver thread */
	char				peer_addr[PEER_ADDR_LEN];
} dpuClient;

static char			   *dpuserv_listen_addr = NULL;
static long				dpuserv_listen_port = -1;
static char			   *dpuserv_base_directory = NULL;
static long				dpuserv_num_workers = -1;
static char			   *dpuserv_identifier = NULL;
static bool				verbose = false;
static pthread_mutex_t	dpu_client_mutex;
static dlist_head		dpu_client_list;
static pthread_mutex_t	dpu_command_mutex;
static pthread_cond_t	dpu_command_cond;
static dlist_head		dpu_command_list;
static volatile bool	got_sigterm = false;

/*
 * getDpuClient
 */
static void
getDpuClient(dpuClient *dclient, int count)
{
	int32_t		refcnt;

	refcnt = __atomic_fetch_add(&dclient->refcnt, count, __ATOMIC_SEQ_CST);
	assert(refcnt > 0);
}

/*
 * putDpuClient
 */
static void
putDpuClient(dpuClient *dclient, int count)
{
	int32_t		refcnt;

	refcnt = __atomic_sub_fetch(&dclient->refcnt, count, __ATOMIC_SEQ_CST);
	assert(refcnt >= 0);
	if (refcnt == 0)
	{
		pthreadMutexLock(&dpu_client_mutex);
		if (dclient->chain.prev && dclient->chain.next)
		{
			dlist_delete(&dclient->chain);
			memset(&dclient->chain, 0, sizeof(dlist_node));
		}
		pthreadMutexUnlock(&dpu_client_mutex);
		
		if (dclient->session)
			free(dclient->session);
		close(dclient->sockfd);
		free(dclient);
	}
}

/*
 * dpuservDpuWorkerMain
 */
static void *
dpuservDpuWorkerMain(void *__priv)
{
	long	worker_id = (long)__priv;

	printf("[DPU-%ld] start\n", worker_id);
	pthreadMutexLock(&dpu_command_mutex);
	while (!got_sigterm)
	{
		if (!dlist_is_empty(&dpu_command_list))
		{
			dlist_node	   *dnode = dlist_pop_head_node(&dpu_command_list);
			XpuCommand	   *xcmd = dlist_container(XpuCommand, chain, dnode);
			dpuClient	   *dclient;
			pthreadMutexUnlock(&dpu_command_mutex);

			dclient = xcmd->priv;
			/*
			 * MEMO: If the least bit of gclient->refcnt is not set,
			 * it means the gpu-client connection is no longer available.
			 * (monitor thread has already gone)
			 */
			if ((dclient->refcnt & 1) != 1)
			{
				switch (xcmd->tag)
				{
					case XpuCommandTag__OpenSession:
					case XpuCommandTag__XpuScanExec:
					default:
						fprintf(stderr, "[DPU-%ld] unknown xPU command (%d)\n",
								worker_id, (int)xcmd->tag);
				}
			}
			if (xcmd)
				free(xcmd);
			putDpuClient(dclient, 2);
			pthreadMutexLock(&dpu_command_mutex);
		}
		else
		{
			pthreadCondWait(&dpu_command_cond,
							&dpu_command_mutex);
		}
	}
	pthreadMutexUnlock(&dpu_command_mutex);
	printf("[worker-%ld] terminate\n", worker_id);
	return NULL;
}

/*
 * dpuservMonitorClient
 */
static void *
__dpuServAllocCommand(void *__priv, size_t sz)
{
	return malloc(sz);
}

static void
__dpuServAttachCommand(void *__priv, XpuCommand *xcmd)
{
	dpuClient  *dclient = (dpuClient *)__priv;

	getDpuClient(dclient, 2);
	xcmd->priv = dclient;

	pthreadMutexLock(&dpu_command_mutex);
	dlist_push_tail(&dpu_command_list, &xcmd->chain);
	pthreadCondSignal(&dpu_command_cond);
	pthreadMutexUnlock(&dpu_command_mutex);
}

TEMPLATE_XPU_CONNECT_RECEIVE_COMMANDS(__dpuServ)

static void *
dpuservMonitorClient(void *__priv)
{
	dpuClient  *dclient = (dpuClient *)__priv;
	
	printf("[%s] %s start\n", dclient->peer_addr, __FUNCTION__);
	while (!got_sigterm)
	{
		struct pollfd  pfd;
		int		rv;

		pfd.fd = dclient->sockfd;
		pfd.events = POLLIN | POLLRDHUP;
		pfd.revents = 0;
		rv = poll(&pfd, 1, -1);
		if (rv < 0)
		{
			if (errno == EINTR)
				continue;
			fprintf(stderr, "failed on poll(2): %m\n");
			break;
		}
		else if (rv > 0)
		{
			assert(rv == 1);
			if (pfd.revents == POLLIN)
			{
				if (__dpuServReceiveCommands(dclient->sockfd, dclient,
											 dclient->peer_addr) < 0)
					break;
			}
			else if (pfd.revents & ~POLLIN)
			{
				fprintf(stderr, "[%s] peer socket closed\n", dclient->peer_addr);
				break;
			}
		}
	}
	putDpuClient(dclient, 1);
	printf("[%s] %s terminated\n", dclient->peer_addr, __FUNCTION__);
	return NULL;
}

static void
dpuserv_signal_handler(int signum)
{
	int		errno_saved = errno;

	if (signum == SIGTERM)
		got_sigterm = true;
	printf("thread %lu got signal %d\n", pthread_self(), signum);
	errno = errno_saved;
}

static int
dpuserv_main(struct sockaddr *addr, socklen_t addr_len)
{
	pthread_t  *dpuserv_workers;
	int			serv_fd;
	int			epoll_fd;
	struct epoll_event epoll_ev;

	/* setup signal handler */
	signal(SIGTERM, dpuserv_signal_handler);
	signal(SIGUSR1, dpuserv_signal_handler);

	/* start worker threads */
	dpuserv_workers = alloca(sizeof(pthread_t) * dpuserv_num_workers);
	for (long i=0; i < dpuserv_num_workers; i++)
	{
		if ((errno = pthread_create(&dpuserv_workers[i], NULL,
									dpuservDpuWorkerMain, (void *)i)) != 0)
			__Elog("failed on pthread_create: %m");
	}
	
	/* setup server socket */
	serv_fd = socket(addr->sa_family, SOCK_STREAM, 0);
	if (serv_fd < 0)
		__Elog("failed on socket(2): %m");
	if (bind(serv_fd, addr, addr_len) != 0)
		__Elog("failed on bind(2): %m");
	if (listen(serv_fd, dpuserv_num_workers) != 0)
		__Elog("failed on listen(2): %m");

	/* setup epoll */
	epoll_fd = epoll_create(1);
	if (epoll_fd < 0)
		__Elog("failed on epoll_create: %m");
	epoll_ev.events = EPOLLIN;
	epoll_ev.data.fd = serv_fd;
	if (epoll_ctl(epoll_fd, EPOLL_CTL_ADD, serv_fd, &epoll_ev) != 0)
		__Elog("failed on epoll_ctl(EPOLL_CTL_ADD): %m");
	epoll_ev.events = EPOLLIN | EPOLLRDHUP;
	epoll_ev.data.fd = fileno(stdin);
	if (epoll_ctl(epoll_fd, EPOLL_CTL_ADD, fileno(stdin), &epoll_ev) != 0)
		__Elog("failed on epoll_ctl(EPOLL_CTL_ADD): %m");
	
	while (!got_sigterm)
	{
		int		rv;

		rv = epoll_wait(epoll_fd, &epoll_ev, 1, 2000);
		if (rv > 0)
		{
			assert(rv == 1);
			if (epoll_ev.data.fd == serv_fd)
			{
				struct sockaddr		peer;
				socklen_t			peer_sz = sizeof(struct sockaddr);
				int					client_fd;

				if ((epoll_ev.events & ~EPOLLIN) != 0)
					__Elog("listen socket raised unexpected error events (%08x): %m",
						   epoll_ev.events);
				
				client_fd = accept(serv_fd, &peer, &peer_sz);
				if (client_fd < 0)
				{
					if (errno != EINTR)
						__Elog("failed on accept: %m");
				}
				else
				{
					dpuClient  *dclient;

					dclient = calloc(1, sizeof(dpuClient));
					if (!dclient)
						__Elog("out of memory: %m");
					dclient->refcnt = 1;
					pthreadMutexInit(&dclient->mutex);
					dclient->sockfd = client_fd;
					if (peer.sa_family == AF_INET)
						inet_ntop(peer.sa_family,
								  (char *)&peer +
								  offsetof(struct sockaddr_in, sin_addr),
								  dclient->peer_addr, PEER_ADDR_LEN);
					else if (peer.sa_family == AF_INET6)
						inet_ntop(peer.sa_family,
								  (char *)&peer +
								  offsetof(struct sockaddr_in6, sin6_addr),
								  dclient->peer_addr, PEER_ADDR_LEN);
					else
						snprintf(dclient->peer_addr, PEER_ADDR_LEN,
								 "Unknown DpuClient");

					pthreadMutexLock(&dpu_client_mutex);
					if ((errno = pthread_create(&dclient->worker, NULL,
												dpuservMonitorClient,
												dclient)) == 0)
					{
						dlist_push_tail(&dpu_client_list,
										&dclient->chain);
					}
					else
					{
						fprintf(stderr, "failed on pthread_create: %m\n");
						close(client_fd);
						free(dclient);
					}
					pthreadMutexUnlock(&dpu_client_mutex);
				}
			}
			else
			{
				char	buffer[1024];
				ssize_t	nbytes;

				assert(epoll_ev.data.fd == fileno(stdin));
				if ((epoll_ev.events & ~EPOLLIN) != 0)
					got_sigterm = true;
				else
				{
					/* make stdin buffer empty */
					nbytes = read(epoll_ev.data.fd, buffer, 1024);
					if (nbytes < 0)
						__Elog("failed on read(stdin): %m");
				}
			}
		}
		else if (rv < 0 && errno != EINTR)
			__Elog("failed on poll(2): %m");
	}
	close(serv_fd);

	/* wait for completion of worker threads */
	pthread_cond_broadcast(&dpu_command_cond);
	for (int i=0; i < dpuserv_num_workers; i++)
		pthread_join(dpuserv_workers[i], NULL);
	pthreadMutexLock(&dpu_client_mutex);
	while (!dlist_is_empty(&dpu_client_list))
	{
		dlist_node *dnode = dlist_pop_head_node(&dpu_client_list);
		dpuClient  *dclient = dlist_container(dpuClient, chain, dnode);

		pthreadMutexUnlock(&dpu_client_mutex);

		pthread_kill(dclient->worker, SIGUSR1);
		pthread_join(dclient->worker, NULL);

		pthreadMutexLock(&dpu_client_mutex);
	}
	pthreadMutexUnlock(&dpu_client_mutex);
	printf("OK terminate\n");
	return 0;
}

int
main(int argc, char *argv[])
{
	static struct option command_options[] = {
		{"addr",       required_argument, 0, 'a'},
		{"port",       required_argument, 0, 'p'},
		{"directory",  required_argument, 0, 'd'},
		{"nworkers",   required_argument, 0, 'n'},
		{"identifier", required_argument, 0, 'i'},
		{"verbose",    no_argument,       0, 'v'},
		{"help",       no_argument,       0, 'h'},
		{NULL, 0, 0, 0},
	};
	struct sockaddr *addr;
	socklen_t	addr_len;

	/* init misc variables */
	pthreadMutexInit(&dpu_client_mutex);
	dlist_init(&dpu_client_list);
	pthreadMutexInit(&dpu_command_mutex);
	pthreadCondInit(&dpu_command_cond);
	dlist_init(&dpu_command_list);

	/* parse command line options */
	for (;;)
	{
		int		c = getopt_long(argc, argv, "a:p:d:n:i:vh", command_options, NULL);
		char   *end;

		if (c < 0)
			break;
		switch (c)
		{
			case 'a':
				if (dpuserv_listen_addr)
					__Elog("-a|--addr option was given twice");
				dpuserv_listen_addr = optarg;
				break;
			
			case 'p':
				if (dpuserv_listen_port > 0)
					__Elog("-p|--port option was given twice");
				dpuserv_listen_port = strtol(optarg, &end, 10);
				if (*optarg == '\0' || *end != '\0')
					__Elog("port number [%s] is not valid", optarg);
				if (dpuserv_listen_port < 1024 || dpuserv_listen_port > USHRT_MAX)
					__Elog("port number [%ld] is out of range", dpuserv_listen_port);
				break;

			case 'd':
				if (dpuserv_base_directory)
					__Elog("-d|--directory option was given twice");
				dpuserv_base_directory = optarg;
				break;
				
			case 'n':
				if (dpuserv_num_workers > 0)
					__Elog("-n|--num-workers option was given twice");
				dpuserv_num_workers = strtol(optarg, &end, 10);
				if (*optarg == '\0' || *end != '\0')
					__Elog("number of workers [%s] is not valid", optarg);
				if (dpuserv_num_workers < 1)
					__Elog("number of workers %ld is out of range",
						   dpuserv_num_workers);
				break;

			case 'i':
				if (dpuserv_identifier)
					__Elog("-i|--identifier option was given twice");
				dpuserv_identifier = optarg;
				break;

			case 'v':
				verbose = true;
				break;
			default:	/* --help */
				fputs("usage: dpuserv [OPTIONS]\n"
					  "\n"
					  "\t-p|--port=PORT           listen port (default: 54321)\n"
					  "\t-d|--directory=DIR       tablespace base (default: .)\n"
					  "\t-n|--nworkers=N_WORKERS  number of workers (default: auto)\n"
					  "\t-i|--identifier=IDENT    security identifier\n"
					  "\t-v|--verbose             verbose output\n"
					  "\t-h|--help                shows this message\n",
					  stderr);
				return 1;
		}
	}
	/* apply default values */
	if (dpuserv_listen_port < 0)
		dpuserv_listen_port = 54321;
	if (!dpuserv_base_directory)
		dpuserv_base_directory = ".";
	if (dpuserv_num_workers < 0)
		dpuserv_num_workers = Max(4 * sysconf(_SC_NPROCESSORS_ONLN), 20);

	/* change the current working directory */
	if (chdir(dpuserv_base_directory) != 0)
		__Elog("failed on chdir('%s'): %m", dpuserv_base_directory);
	/* resolve host and port */
	if (!dpuserv_listen_addr)
	{
		static struct sockaddr_in __addr;

		memset(&__addr, 0, sizeof(__addr));
		__addr.sin_family = AF_INET;
		__addr.sin_port = htons(dpuserv_listen_port);
		__addr.sin_addr.s_addr = htonl(INADDR_ANY);
		addr = (struct sockaddr *)&__addr;
		addr_len = sizeof(__addr);
	}
	else
	{
		struct addrinfo hints;
		struct addrinfo *res;
		char		temp[50];

		memset(&hints, 0, sizeof(struct addrinfo));
		hints.ai_family = AF_UNSPEC;
		hints.ai_socktype = SOCK_STREAM;
		snprintf(temp, sizeof(temp), "%ld", dpuserv_listen_port);
		if (getaddrinfo(dpuserv_listen_addr, temp, &hints, &res) != 0)
			__Elog("failed on getaddrinfo('%s',%ld): %m",
				   dpuserv_listen_addr,
				   dpuserv_listen_port);
		addr = res->ai_addr;
		addr_len = res->ai_addrlen;
	}
	return dpuserv_main(addr, addr_len);
}
