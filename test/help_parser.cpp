#include <iostream>
#include <regex>
#include <string>
#include <map>

struct Lexer {
    std::string* text;
    size_t tot;
    size_t pos;
    int short_opt_spaces;
    int long_opt_spaces;
    // items of an option
    // 1. <string> short form
    // 2. <string> long form
    // 3. <string> option form
    // 4. <string> description
    std::vector<std::vector<std::string>> dict{};
    Lexer(std::string* t): text(t) {
        pos = 0;
        tot = t->size();
        short_opt_spaces = long_opt_spaces = -1;
    }
    int nextIsOption() {
        if (pos + 1 < tot && text->at(pos) == '-' && !isspace(text->at(pos + 1)) && text->at(pos + 1) != '-')
            return 1;
        if (pos + 2 < tot && text->at(pos) == '-' && text->at(pos + 1) == '-' && !isspace(text->at(pos + 2)))
            return 2;
        return 0;
    }
    void leapSpace() {
        while (pos < tot && isspace(text->at(pos)))
            ++pos;
    }
    int countSpaces() {
        int spaces = 0;
        while (pos < tot && isspace(text->at(pos))) {
            if (text->at(pos) == '\n') {
                spaces = 0;
            } else {
                ++spaces;
            }
            ++pos;
        }
        return spaces;
    }
    void shortOption(std::string &short_opt) {
        size_t start = pos;
        while (pos < tot && !isspace(text->at(pos)) && text->at(pos) != ',')
            ++pos;
        size_t len = pos - start;
        short_opt = text->substr(start, len);
    }
    void longOption(std::string &long_opt) {
        size_t start = pos;
        while (pos < tot && !isspace(text->at(pos)))
            ++pos;
        size_t len = pos - start;
        long_opt = text->substr(start, len);
    }
    void description(std::string &descript) {
        size_t start;
        leapSpace();
        descript += " ";
        start = pos;
        while (true) {
            // read until the end of this line
            while (pos < tot && text->at(pos) != '\n')
                ++pos;
            ++pos;
            descript += text->substr(start, pos - start);
            // if the following line is an empty line, we consider this item as finished
            if (pos < tot && text->at(pos) == '\n')
                break;
            int spaces = countSpaces();
            // we reach the end of text
            if (pos >= tot)
                break;
            int next_option = nextIsOption();
            // we reach the next item
            if (next_option == 1) {
                if (short_opt_spaces < 0)
                    short_opt_spaces = spaces;
                if (spaces <= short_opt_spaces)
                    break;
            } else if (next_option == 2) {
                if (long_opt_spaces < 0)
                    long_opt_spaces = spaces;
                if (spaces <= long_opt_spaces)
                    break;
            }
            start = pos;
        }
        descript.erase(std::find_if(descript.rbegin(), descript.rend(), [](unsigned char ch) {
            return !std::isspace(ch);
        }).base(), descript.end());
        // if the following line is an empty line, try to get another item below it
        if (pos < tot && text->at(pos) == '\n')
            leapSpace();
    }
    void parse() {
        bool flag = true;
        int spaces = 0;
        while(pos < tot) {
            int next = nextIsOption();
            if (flag && next > 0) {
                if (next == 1)
                    short_opt_spaces = spaces;
                else
                    long_opt_spaces = spaces;
                break;
            }
            if (text->at(pos) == '\n') {
                flag = true;
                spaces = 0;
            } else if (!isspace(text->at(pos))) {
                flag = false;
            } else {
                ++spaces;
            }
            ++pos;
        }
        std::string short_opt, long_opt, opt, descript;
        while (pos < tot) {
            if (!nextIsOption())
                break;
            descript.clear();
            if (text->at(pos + 1) != '-') {
                shortOption(short_opt);
                descript += short_opt;
                long_opt.clear();
                if (pos < tot && text->at(pos) == ',') {
                    ++pos;
                    leapSpace();
                    if (nextIsOption() == 2)
                        longOption(long_opt);
                    descript += ", " + long_opt;
                }
            } else {
                short_opt.clear();
                longOption(long_opt);
                descript += long_opt;
            }
            description(descript);
            // option form
            if (long_opt.size() > 0) {
                opt = long_opt;
                int param_pos = long_opt.find_first_of('=');
                if (param_pos != std::string::npos) {
                    opt = opt.substr(0, param_pos + 1);
                    if (long_opt[long_opt.size() - 1] == ']')
                        opt += ']';
                }
            } else {
                opt = short_opt;
            }
            dict.push_back(std::vector<std::string>({short_opt, long_opt, opt, descript}));
        }
    }
    // TODO(@seeeagull): for debugging
    void printDict() {
        int i = 1;
        for (auto &it: dict) {
            std::cout << "#" << i << ":" << std::endl;
            std::cout << "  short form:  " << it[0] << std::endl;
            std::cout << "  long form:   " << it[1] << std::endl;
            std::cout << "  option form: " << it[2] << std::endl;
            std::cout << "  description:\n" << it[3] << std::endl;
            ++i;
        }
    }
};

int main() {
    std::string s = "Syntax: ./InstantMeshes [options] <input mesh / point cloud / application state snapshot>\n"
                    "Options:\n"
                    "   -o, --output <output>     Writes to the specified PLY/OBJ output file in batch mode\n"
                    "   -t, --threads <count>     Number of threads used for parallel computations\n"
                    "   -d, --deterministic       Prefer (slower) deterministic algorithms\n"
                    "   -c, --crease <degrees>    Dihedral angle threshold for creases\n"
                    "   -S, --smooth <iter>       Number of smoothing & ray tracing reprojection steps (default: 2)\n"
                    "   -D, --dominant            Generate a tri/quad dominant mesh instead of a pure tri/quad mesh\n"
                    "   -i, --intrinsic           Intrinsic mode (extrinsic is the default)\n"
                    "   -b, --boundaries          Align to boundaries (only applies when the mesh is not closed)\n"
                    "   -r, --rosy <number>       Specifies the orientation symmetry type (2, 4, or 6)\n"
                    "   -p, --posy <number>       Specifies the position symmetry type (4 or 6)\n"
                    "   -s, --scale <scale>       Desired world space length of edges in the output\n"
                    "   -f, --faces <count>       Desired face count of the output mesh\n"
                    "   -v, --vertices <count>    Desired vertex count of the output mesh\n"
                    "   -C, --compat              Compatibility mode to load snapshots from old software versions\n"
                    "   -k, --knn <count>         Point cloud mode: number of adjacent points to consider\n"
                    "   -F, --fullscreen          Open a full-screen window\n"
                    "   -h, --help                Display this message";
    
    // s = "RobustTetMeshing\n"
    //     "Usage: ./TetWild [OPTIONS] input [output]\n\n"
    //     "Positionals:\n"
    //     "input TEXT REQUIRED         Input surface mesh INPUT in .off/.obj/.stl/.ply format. (string, required)\n"
    //     "output TEXT                 Output tetmesh OUTPUT in .msh or .mesh format. (string, optional, default: input_file+postfix+'.msh')\n\n"
    //     "Options:\n"
    //     "-h,--help                   Print this help message and exit\n"
    //     "--input TEXT REQUIRED       Input surface mesh INPUT in .off/.obj/.stl/.ply format. (string, required)\n"
    //     "--output TEXT               Output tetmesh OUTPUT in .msh or .mesh format. (string, optional, default: input_file+postfix+'.msh')\n"
    //     "--postfix TEXT              Postfix P for output files. (string, optional, default: '_')\n"
    //     "-a,--ideal-absolute-edge-length FLOAT Excludes: --ideal-edge-length\n"
    //     "                            Absolute edge length (not scaled by bbox). -a and -l cannot both be given as arguments.\n"
    //     "-l,--ideal-edge-length FLOAT Excludes: --ideal-absolute-edge-length\n"
    //     "                            ideal_edge_length = diag_of_bbox * L. (double, optional, default: 0.05)\n"
    //     "-e,--epsilon FLOAT          epsilon = diag_of_bbox * EPS. (double, optional, default: 1e-3)\n"
    //     "--stage INT                 Run pipeline in stage STAGE. (integer, optional, default: 1)\n"
    //     "--filter-energy FLOAT       Stop mesh improvement when the maximum energy is smaller than ENERGY. (double, optional, default: 10)\n"
    //     "--max-pass INT              Do PASS mesh improvement passes in maximum. (integer, optional, default: 80)\n"
    //     "--targeted-num-v INT        Output tetmesh that contains TV vertices. (integer, optional, tolerance: 5%)\n"
    //     "--bg-mesh TEXT              Background tetmesh BGMESH in .msh format for applying sizing field. (string, optional)\n"
    //     "--log TEXT                  Log info to given file.\n"
    //     "--level INT                 Log level (0 = most verbose, 6 = off).\n"
    //     "--save-mid-result INT       Get result without winding number: --save-mid-result 2\n"
    //     "--no-voxel                  Use voxel stuffing before BSP subdivision.\n"
    //     "--is-laplacian              Do Laplacian smoothing for the surface of output on the holes of input (optional)\n"
    //     "-q,--is-quiet               Mute console output. (optional)";

    s = "Usage: ls [OPTION]... [FILE]...\n"
        "List information about the FILEs (the current directory by default).\n"
        "Sort entries alphabetically if none of -cftuvSUX nor --sort is specified.\n\n"
        "Mandatory arguments to long options are mandatory for short options too.\n"
        "  -a, --all                  do not ignore entries starting with .\n"
        "  -A, --almost-all           do not list implied . and ..\n"
        "      --author               with -l, print the author of each file\n"
        "  -b, --escape               print C-style escapes for nongraphic characters\n"
        "      --block-size=SIZE      with -l, scale sizes by SIZE when printing them;\n"
        "                               e.g., '--block-size=M'; see SIZE format below\n"
        "  -B, --ignore-backups       do not list implied entries ending with ~\n"
        "  -c                         with -lt: sort by, and show, ctime (time of last\n"
        "                               modification of file status information);\n"
        "                               with -l: show ctime and sort by name;\n"
        "                               otherwise: sort by ctime, newest first\n"
        "  -C                         list entries by columns\n"
        "      --color[=WHEN]         colorize the output; WHEN can be 'always' (default\n"
        "                               if omitted), 'auto', or 'never'; more info below\n"
        "  -d, --directory            list directories themselves, not their contents\n"
        "  -D, --dired                generate output designed for Emacs' dired mode\n"
        "  -f                         do not sort, enable -aU, disable -ls --color\n"
        "  -F, --classify             append indicator (one of */=>@|) to entries\n"
        "      --file-type            likewise, except do not append '*'\n"
        "      --format=WORD          across -x, commas -m, horizontal -x, long -l,\n"
        "                               single-column -1, verbose -l, vertical -C\n"
        "      --full-time            like -l --time-style=full-iso\n"
        "  -g                         like -l, but do not list owner\n"
        "      --group-directories-first\n"
        "                             group directories before files;\n"
        "                               can be augmented with a --sort option, but any\n"
        "                              use of --sort=none (-U) disables grouping\n"
        "  -G, --no-group             in a long listing, don\'t print group names\n"
        "  -h, --human-readable       with -l and -s, print sizes like 1K 234M 2G etc.\n"
        "      --si                   likewise, but use powers of 1000 not 1024\n"
        "  -H, --dereference-command-line\n"
        "                             follow symbolic links listed on the command line\n"
        "      --dereference-command-line-symlink-to-dir\n"
        "                             follow each command line symbolic link\n"
        "                               that points to a directory\n"
        "      --hide=PATTERN         do not list implied entries matching shell PATTERN\n"
        "                               (overridden by -a or -A)\n"
        "      --hyperlink[=WHEN]     hyperlink file names; WHEN can be \'always\'\n"
        "                               (default if omitted), \'auto\', or \'never\'\n"
        "      --indicator-style=WORD  append indicator with style WORD to entry names:\n"
        "                               none (default), slash (-p),\n"
        "                               file-type (--file-type), classify (-F)\n"
        "  -i, --inode                print the index number of each file\n"
        "  -I, --ignore=PATTERN       do not list implied entries matching shell PATTERN\n"
        "  -k, --kibibytes            default to 1024-byte blocks for disk usage;\n"
        "                               used only with -s and per directory totals\n"
        "  -l                         use a long listing format\n"
        "  -L, --dereference          when showing file information for a symbolic\n"
        "                               link, show information for the file the link\n"
        "                               references rather than for the link itself\n"
        "  -m                         fill width with a comma separated list of entries\n"
        "  -n, --numeric-uid-gid      like -l, but list numeric user and group IDs\n"
        "  -N, --literal              print entry names without quoting\n"
        "  -o                         like -l, but do not list group information\n"
        "  -p, --indicator-style=slash\n"
        "                             append / indicator to directories\n"
        "  -q, --hide-control-chars   print ? instead of nongraphic characters\n"
        "      --show-control-chars   show nongraphic characters as-is (the default,\n"
        "                               unless program is \'ls\' and output is a terminal)\n"
        "  -Q, --quote-name           enclose entry names in double quotes\n"
        "      --quoting-style=WORD   use quoting style WORD for entry names:\n"
        "                               literal, locale, shell, shell-always,\n"
        "                               shell-escape, shell-escape-always, c, escape\n"
        "                               (overrides QUOTING_STYLE environment variable)\n"
        "  -r, --reverse              reverse order while sorting\n"
        "  -R, --recursive            list subdirectories recursively\n"
        "  -s, --size                 print the allocated size of each file, in blocks\n"
        "  -S                         sort by file size, largest first\n"
        "      --sort=WORD            sort by WORD instead of name: none (-U), size (-S),\n"
        "                               time (-t), version (-v), extension (-X)\n"
        "      --time=WORD            change the default of using modification times;\n"
        "                               access time (-u): atime, access, use;\n"
        "                               change time (-c): ctime, status;\n"
        "                               birth time: birth, creation;\n"
        "                             with -l, WORD determines which time to show;\n"
        "                             with --sort=time, sort by WORD (newest first)\n"
        "      --time-style=TIME_STYLE  time/date format with -l; see TIME_STYLE below\n"
        "  -t                         sort by time, newest first; see --time\n"
        "  -T, --tabsize=COLS         assume tab stops at each COLS instead of 8\n"
        "  -u                         with -lt: sort by, and show, access time;\n"
        "                               with -l: show access time and sort by name;\n"
        "                               otherwise: sort by access time, newest first\n"
        "  -U                         do not sort; list entries in directory order\n"
        "  -v                         natural sort of (version) numbers within text\n"
        "  -w, --width=COLS           set output width to COLS.  0 means no limit\n"
        "  -x                         list entries by lines instead of by columns\n"
        "  -X                         sort alphabetically by entry extension\n"
        "  -Z, --context              print any security context of each file\n"
        "  -1                         list one file per line.  Avoid \'\\n\' with -q or -b\n"
        "      --help     display this help and exit\n"
        "      --version  output version information and exit\n\n"
        "The SIZE argument is an integer and optional unit (example: 10K is 10*1024).\n"
        "Units are K,M,G,T,P,E,Z,Y (powers of 1024) or KB,MB,... (powers of 1000).\n"
        "Binary prefixes can be used, too: KiB=K, MiB=M, and so on.\n\n"
        "The TIME_STYLE argument can be full-iso, long-iso, iso, locale, or +FORMAT.\n"
        "FORMAT is interpreted like in date(1).  If FORMAT is FORMAT1<newline>FORMAT2,\n"
        "then FORMAT1 applies to non-recent files and FORMAT2 to recent files.\n"
        "TIME_STYLE prefixed with \'posix-\' takes effect only outside the POSIX locale.\n"
        "Also the TIME_STYLE environment variable sets the default style to use.\n\n"
        "Using color to distinguish file types is disabled both by default and\n"
        "with --color=never.  With --color=auto, ls emits color codes only when\n"
        "standard output is connected to a terminal.  The LS_COLORS environment\n"
        "variable can change the settings.  Use the dircolors command to set it.\n\n"
        "Exit status:\n"
        " 0  if OK,\n"
        " 1  if minor problems (e.g., cannot access subdirectory),\n"
        " 2  if serious trouble (e.g., cannot access command-line argument).\n\n"
        "GNU coreutils online help: <https://www.gnu.org/software/coreutils/>\n"
        "Full documentation <https://www.gnu.org/software/coreutils/ls>\n"
        "or available locally via: info \'(coreutils) ls invocation\'</pre>";

    // s = "usage: python3 [option] ... [-c cmd | -m mod | file | -] [arg] ...\n"
    //     "Options and arguments (and corresponding environment variables):\n"
    //     "-b     : issue warnings about str(bytes_instance), str(bytearray_instance)\n"
    //     "        and comparing bytes/bytearray with str. (-bb: issue errors)\n"
    //     "-B     : don't write .pyc files on import; also PYTHONDONTWRITEBYTECODE=x\n"
    //     "-c cmd : program passed in as string (terminates option list)\n"
    //     "-d     : turn on parser debugging output (for experts only, only works on\n"
    //     "        debug builds); also PYTHONDEBUG=x\n"
    //     "-E     : ignore PYTHON* environment variables (such as PYTHONPATH)\n"
    //     "-h     : print this help message and exit (also -? or --help)\n"
    //     "-i     : inspect interactively after running script; forces a prompt even\n"
    //     "        if stdin does not appear to be a terminal; also PYTHONINSPECT=x\n"
    //     "-I     : isolate Python from the user's environment (implies -E and -s)\n"
    //     "-m mod : run library module as a script (terminates option list)\n"
    //     "-O     : remove assert and __debug__-dependent statements; add .opt-1 before\n"
    //     "        .pyc extension; also PYTHONOPTIMIZE=x\n"
    //     "-OO    : do -O changes and also discard docstrings; add .opt-2 before\n"
    //     "        .pyc extension\n"
    //     "-q     : don't print version and copyright messages on interactive startup\n"
    //     "-s     : don't add user site directory to sys.path; also PYTHONNOUSERSITE\n"
    //     "-S     : don't imply 'import site' on initialization\n"
    //     "-u     : force the stdout and stderr streams to be unbuffered;\n"
    //     "        this option has no effect on stdin; also PYTHONUNBUFFERED=x\n"
    //     "-v     : verbose (trace import statements); also PYTHONVERBOSE=x\n"
    //     "        can be supplied multiple times to increase verbosity\n"
    //     "-V     : print the Python version number and exit (also --version)\n"
    //     "        when given twice, print more information about the build\n"
    //     "-W arg : warning control; arg is action:message:category:module:lineno\n"
    //     "        also PYTHONWARNINGS=arg\n"
    //     "-x     : skip first line of source, allowing use of non-Unix forms of #!cmd\n"
    //     "-X opt : set implementation-specific option. The following options are available:\n"
    //     "        -X faulthandler: enable faulthandler\n"
    //     "        -X showrefcount: output the total reference count and number of used\n"
    //     "            memory blocks when the program finishes or after each statement in the\n"
    //     "            interactive interpreter. This only works on debug builds\n"
    //     "        -X tracemalloc: start tracing Python memory allocations using the\n"
    //     "            tracemalloc module. By default, only the most recent frame is stored in a\n"
    //     "            traceback of a trace. Use -X tracemalloc=NFRAME to start tracing with a\n"
    //     "            traceback limit of NFRAME frames\n"
    //     "        -X importtime: show how long each import takes. It shows module name,\n"
    //     "            cumulative time (including nested imports) and self time (excluding\n"
    //     "            nested imports). Note that its output may be broken in multi-threaded\n"
    //     "            application. Typical usage is python3 -X importtime -c 'import asyncio'\n"
    //     "        -X dev: enable CPython's \"development mode\", introducing additional runtime\n"
    //     "            checks which are too expensive to be enabled by default. Effect of the\n"
    //     "            developer mode:\n"
    //     "                * Add default warning filter, as -W default\n"
    //     "                * Install debug hooks on memory allocators: see the PyMem_SetupDebugHooks()\n"
    //     "                C function\n"
    //     "                * Enable the faulthandler module to dump the Python traceback on a crash\n"
    //     "                * Enable asyncio debug mode\n"
    //     "                * Set the dev_mode attribute of sys.flags to True\n"
    //     "                * io.IOBase destructor logs close() exceptions\n"
    //     "        -X utf8: enable UTF-8 mode for operating system interfaces, overriding the default\n"
    //     "            locale-aware mode. -X utf8=0 explicitly disables UTF-8 mode (even when it would\n"
    //     "            otherwise activate automatically)\n"
    //     "        -X pycache_prefix=PATH: enable writing .pyc files to a parallel tree rooted at the\n"
    //     "            given directory instead of to the code tree\n"
    //     "        -X warn_default_encoding: enable opt-in EncodingWarning for 'encoding=None'\n"
    //     "        -X int_max_str_digits=number: limit the size of int<->str conversions.\n"
    //     "            This helps avoid denial of service attacks when parsing untrusted data.\n"
    //     "            The default is sys.int_info.default_max_str_digits.  0 disables.\n\n"
    //     "--check-hash-based-pycs always|default|never:\n"
    //     "    control how Python invalidates hash-based .pyc files\n"
    //     "file   : program read from script file\n"
    //     "-      : program read from stdin (default; interactive mode if a tty)\n"
    //     "arg ...: arguments passed to program in sys.argv[1:]\n\n"
    //     "Other environment variables:\n"
    //     "PYTHONSTARTUP: file executed on interactive startup (no default)\n"
    //     "PYTHONPATH   : ':'-separated list of directories prefixed to the\n"
    //     "            default module search path.  The result is sys.path.\n"
    //     "PYTHONHOME   : alternate <prefix> directory (or <prefix>:<exec_prefix>).\n"
    //     "            The default module search path uses <prefix>/lib/pythonX.X.\n"
    //     "PYTHONPLATLIBDIR : override sys.platlibdir.\n"
    //     "PYTHONCASEOK : ignore case in 'import' statements (Windows).\n"
    //     "PYTHONUTF8: if set to 1, enable the UTF-8 mode.\n"
    //     "PYTHONIOENCODING: Encoding[:errors] used for stdin/stdout/stderr.\n"
    //     "PYTHONFAULTHANDLER: dump the Python traceback on fatal errors.\n"
    //     "PYTHONHASHSEED: if this variable is set to 'random', a random value is used\n"
    //     "to seed the hashes of str and bytes objects.  It can also be set to an\n"
    //     "integer in the range [0,4294967295] to get hash values with a\n"
    //     "predictable seed.\n"
    //     "PYTHONINTMAXSTRDIGITS: limits the maximum digit characters in an int value\n"
    //     "when converting from a string and when converting an int back to a str.\n"
    //     "A value of 0 disables the limit.  Conversions to or from bases 2, 4, 8,\n"
    //     "16, and 32 are never limited.\n"
    //     "PYTHONMALLOC: set the Python memory allocators and/or install debug hooks\n"
    //     "on Python memory allocators. Use PYTHONMALLOC=debug to install debug\n"
    //     "hooks.\n"
    //     "PYTHONCOERCECLOCALE: if this variable is set to 0, it disables the locale\n"
    //     "coercion behavior. Use PYTHONCOERCECLOCALE=warn to request display of\n"
    //     "locale coercion and locale compatibility warnings on stderr.\n"
    //     "PYTHONBREAKPOINT: if this variable is set to 0, it disables the default\n"
    //     "debugger. It can be set to the callable of your debugger of choice.\n"
    //     "PYTHONDEVMODE: enable the development mode.\n"
    //     "PYTHONPYCACHEPREFIX: root directory for bytecode cache (pyc) files.\n"
    //     "PYTHONWARNDEFAULTENCODING: enable opt-in EncodingWarning for 'encoding=None'.\n";
    Lexer lexer(&s);
    lexer.parse();
    lexer.printDict();
    return 0;
}