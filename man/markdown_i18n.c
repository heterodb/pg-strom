#include <ctype.h>
#include <libgen.h>
#include <stdio.h>
#include <strings.h>
#include <unistd.h>

static const char *known_langs[] = {
	"en",	/* default */
	"ja",
	"zh",
	"de",
	"fr",
	"ru",
	"es",
	"pt",
	NULL
};

static void
parse_i18n(FILE *filp, FILE *fout, const char *lang)
{
	char	buf[20];
	int		depth = 0;
	int		valid_block = 1;
	int		meet_newline = 1;
	int		lineno = 1;
	int		i, c;

	while ((c = fgetc(filp)) != EOF)
	{
	restart:
		if (c == '@')
		{
			for (i=0; i < sizeof(buf) - 1; i++)
			{
				c = fgetc(filp);
				if (isalpha(c) || c == '-')
					buf[i] = c;
				else if (c == ':')
				{
					int		valid_line;

					if (depth > 0)
					{
						fprintf(stderr, "%d: cannot use i18n line in block\n", lineno);
						break;
					}
					if (!meet_newline)
					{
						fprintf(stderr, "%d: i18n line is available only line-head\n", lineno);
						break;
					}
					buf[i] = '\0';
					valid_line = (strcasecmp(lang, buf) == 0);
					meet_newline = 0;
					while ((c = fgetc(filp)) != EOF)
					{
						if ((!meet_newline && (c == '\r' || c == '\n')) ||
							(meet_newline == '\r' && c == '\n'))
							meet_newline = c;
						else if (meet_newline)
							goto restart;
						if (valid_line)
							fputc(c, fout);
					}
					return;		/* EOF */
				}
				else if (c == '{')
				{
					if (depth > 0)
					{
						fprintf(stderr, "%d: cannot nest i18n blocks\n",
								lineno);
						break;
					}
					buf[i] = '\0';
					/* begin i18n block */
					valid_block = (strcasecmp(lang, buf) == 0);
					depth = 1;
					goto next_char;
				}
				else
					break;
			}
			/* it was not a control token */
			if (valid_block)
			{
				buf[i] = '\0';
				fprintf(fout, "@%s", buf);
			}
		}
		else if (depth > 0)
		{
			if (c == '{')
				depth++;
			else if ((!meet_newline && (c == '\n' || c == '\r')) ||
					 (meet_newline == '\r' && c == '\n'))
				meet_newline = c;
			else if (meet_newline)
			{
				if (c == '\n' || c == '\r')
					meet_newline = c;
				else
					meet_newline = 0;
				lineno++;
			}
			if (c == '}' && --depth == 0)
				valid_block = 1;
			else if (valid_block)
				fputc(c, fout);
		}
		else
		{
			if ((!meet_newline && (c == '\n' || c == '\r')) ||
				(meet_newline == '\r' && c == '\n'))
				meet_newline = c;
			else if (meet_newline)
			{
				if (c == '\n' || c == '\r')
                    meet_newline = c;
				else
					meet_newline = 0;
				lineno++;
			}
			fputc(c, fout);
		}
	next_char:
		;
	}
}

int main(int argc, char *argv[])
{
	const char *lang = NULL;
	const char *filename = NULL;
	const char *outfile = NULL;
	int		i, c;
	FILE   *filp;
	FILE   *fout = stdout;

	while ((c = getopt(argc, argv, "f:l:o:h")) != -1)
	{
		switch (c)
		{
			case 'f':
				if (filename != NULL)
				{
					fputs("-f was specified twice", stderr);
					return 1;
				}
				filename = optarg;
				break;
			case 'l':
				if (lang)
				{
					fputs("-l was specified twice", stderr);
					return 1;
				}

				for (i=0; known_langs[i] != NULL; i++)
				{
					if (strcasecmp(optarg, known_langs[i]) == 0)
					{
						lang = known_langs[i];
						break;
					}
				}
				if (!lang)
				{
					fprintf(stderr, "not known language code: %s", optarg);
					return 1;
				}
				break;
			case 'o':
				if (outfile != NULL)
				{
					fputs("-o was specified twice", stderr);
					return 1;
				}
				outfile = optarg;
				break;
			case 'h':
			default:
				fprintf(stderr, "usage: %s [-f <filename>][-l <lang>]\n",
						basename(argv[0]));
				return 1;
		}
	}

	/* default: stdin */
	if (!filename)
		filp = stdin;
	else
	{
		filp = fopen(filename, "rb");
		if (!filp)
		{
			fprintf(stderr, "failed on fopen('%s'): %m", filename);
			return 1;
		}
	}

	/* default: en */
	if (!lang)
		lang = known_langs[0];

	/* default: stdout */
	if (!outfile)
		fout = stdout;
	else
	{
		fout = fopen(outfile, "wb");
		if (!fout)
		{
			fprintf(stderr, "failed on fopen('%s'): %m", outfile);
			return 1;
		}
	}
	parse_i18n(filp, fout, lang);

	if (filename)
		fclose(filp);

	return 0;
}
