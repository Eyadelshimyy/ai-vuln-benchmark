"""Per-language security sources, sinks, and sanitizers.

Looked up by language id. The taint engine never branches on language names.
"""

from __future__ import annotations

from app.domain.security import VulnerabilityClass
from app.security.definitions import (
    LanguageSecurityVocab,
    SanitizerDefinition,
    SinkCertainty,
    SinkDefinition,
    SourceDefinition,
)

_WEAK_CRYPTO = ("md5", "sha1", "DES", "RC4", "ECB")


def _src(source_id: str, *patterns: str, kind: str = "http") -> SourceDefinition:
    return SourceDefinition(source_id=source_id, patterns=tuple(patterns), kind=kind)


def _sink(
    sink_id: str,
    vuln: VulnerabilityClass,
    *names: str,
    certainty: SinkCertainty = SinkCertainty.SENSITIVE,
    alternatives: tuple[str, ...] = (),
    condition: str = "attacker-controlled data reaches this API",
    notes: str = "",
    argument_index: int = 0,
    argument_indexes: tuple[int, ...] = (),
    required_context: str = "",
    sanitizer_kinds: tuple[str, ...] = (),
) -> SinkDefinition:
    if vuln is SQL and not argument_indexes:
        argument_indexes = (0,)
    return SinkDefinition(
        sink_id=sink_id,
        vulnerability_class=vuln,
        api_names=tuple(names),
        certainty=certainty,
        safe_alternatives=alternatives,
        dangerous_condition=condition,
        notes=notes,
        qualified_substrings=tuple(names),
        argument_index=argument_index,
        argument_indexes=argument_indexes,
        required_context=required_context,
        sanitizer_kinds=sanitizer_kinds,
    )


def _san(sanitizer_id: str, *names: str, kind: str, effective: bool = False) -> SanitizerDefinition:
    return SanitizerDefinition(
        sanitizer_id=sanitizer_id,
        api_names=tuple(names),
        kind=kind,
        effective=effective,
    )


SQL = VulnerabilityClass.SQL_INJECTION
CMD = VulnerabilityClass.COMMAND_INJECTION
PATH = VulnerabilityClass.POTENTIAL_PATH_TRAVERSAL
SSRF = VulnerabilityClass.SSRF
XSS = VulnerabilityClass.XSS
DESER = VulnerabilityClass.UNSAFE_DESERIALIZATION
EVAL = VulnerabilityClass.DYNAMIC_EXECUTION
REDIR = VulnerabilityClass.UNSAFE_REDIRECT
PDESER = VulnerabilityClass.POTENTIAL_UNSAFE_DESERIALIZATION

PYTHON = LanguageSecurityVocab(
    language_id="python",
    sources=(
        _src(
            "py.http",
            "request.args",
            "request.form",
            "request.json",
            "request.data",
            "request.GET",
            "request.POST",
            "request.query_params",
            "request.path_params",
            "request.cookies",
            "flask.request",
            kind="http",
        ),
        _src("py.cli", "sys.argv", "input", kind="cli"),
        _src("py.env", "os.environ", "getenv", kind="env"),
    ),
    sinks=(
        _sink(
            "py.sql",
            SQL,
            "execute",
            "executemany",
            "raw",
            "cursor.execute",
            "objects.extra",
            "RawSQL",
            alternatives=("parameterized execute", "QuerySet.filter"),
        ),
        _sink(
            "py.cmd",
            CMD,
            "os.system",
            "os.popen",
            "subprocess.call",
            "subprocess.run",
            "subprocess.Popen",
            "subprocess.check_output",
            alternatives=("subprocess.run with argv list",),
        ),
        _sink(
            "py.path",
            PATH,
            "open",
            "Path",
            "send_file",
            "sendfile",
            condition="user-controlled path reaches a file API",
        ),
        _sink(
            "py.ssrf",
            SSRF,
            "requests.get",
            "requests.post",
            "requests.put",
            "requests.delete",
            "requests.request",
            "urlopen",
            "httpx.get",
            "httpx.post",
            "httpx.request",
        ),
        _sink("py.xss", XSS, "Markup", "mark_safe", "html_safe", "jinja2.Markup"),
        _sink("py.deser", DESER, "pickle.loads", "yaml.load", "marshal.loads", "shelve.open"),
        _sink("py.eval", EVAL, "eval", "exec", "compile", "render_template_string"),
        _sink("py.redir", REDIR, "redirect", "HttpResponseRedirect"),
    ),
    sanitizers=(
        _san(
            "py.html",
            "html.escape",
            "markupsafe.escape",
            kind="html_encode",
            effective=True,
        ),
        _san("py.sql", "execute", kind="sql_parameterize"),
        _san("py.path", "os.path.realpath", "Path.resolve", kind="path_canonicalize"),
        _san("py.shell", "shlex.quote", kind="shell_escape", effective=True),
        _san("py.url", "url_has_allowed_host_and_scheme", kind="url_allowlist", effective=True),
    ),
    crypto_names=_WEAK_CRYPTO,
    extra_sources_by_framework={
        "django": (_src("django.http", "request.GET", "request.POST", "request.META"),),
        "flask": (_src("flask.http", "request.args", "request.form", "request.values"),),
        "fastapi": (_src("fastapi.http", "Query", "Path", "Body"),),
    },
)

JAVASCRIPT = LanguageSecurityVocab(
    language_id="javascript",
    sources=(
        _src(
            "js.http",
            "req.query",
            "req.body",
            "req.params",
            "req.headers",
            "req.cookies",
            "request.query",
            "request.body",
            "request.params",
            kind="http",
        ),
        _src("js.cli", "process.argv", kind="cli"),
        _src("js.env", "process.env", kind="env"),
        _src(
            "js.browser", "window.location", "location.search", "document.location", kind="browser"
        ),
        _src("js.nest", "Query", "Param", "Body", "Headers", kind="http"),
    ),
    sinks=(
        _sink(
            "js.sql",
            SQL,
            "query",
            "execute",
            "sequelize.query",
            "knex.raw",
            "raw",
            alternatives=("bound parameters",),
        ),
        _sink(
            "js.cmd",
            CMD,
            "exec",
            "execSync",
            "spawn",
            "spawnSync",
            "child_process.exec",
            "execFile",
            argument_indexes=(0, 1),
        ),
        _sink(
            "js.path", PATH, "readFile", "readFileSync", "writeFile", "createReadStream", "sendFile"
        ),
        _sink(
            "js.ssrf", SSRF, "fetch", "axios.get", "axios.post", "axios.request", "http.get", "got"
        ),
        _sink("js.xss", XSS, "innerHTML", "outerHTML", "document.write", "dangerouslySetInnerHTML"),
        _sink(
            "js.json",
            PDESER,
            "JSON.parse",
            certainty=SinkCertainty.INDICATOR,
            notes="JSON.parse is not code execution; flagged only as a potential untrusted-data indicator.",
        ),
        _sink("js.deser", DESER, "unserialize", "node-serialize"),
        _sink("js.eval", EVAL, "eval", "Function", "runInThisContext", "runInNewContext"),
        _sink("js.redir", REDIR, "redirect", "NextResponse.redirect"),
    ),
    sanitizers=(
        _san(
            "js.html",
            "escapeHtml",
            "encodeURIComponent",
            "DOMPurify.sanitize",
            kind="html_encode",
            effective=True,
        ),
        _san("js.path", "path.normalize", "path.resolve", "path.resolve", kind="path_canonicalize"),
        _san("js.sql", "mysql.format", kind="sql_parameterize"),
    ),
    crypto_names=_WEAK_CRYPTO,
    extra_sources_by_framework={
        "express": (_src("express.http", "req.query", "req.body", "req.params"),),
        "next.js": (_src("next.http", "searchParams", "req.query"),),
        "nestjs": (_src("nest.http", "Query", "Param", "Body"),),
    },
)

TYPESCRIPT = LanguageSecurityVocab(
    language_id="typescript",
    sources=JAVASCRIPT.sources,
    sinks=JAVASCRIPT.sinks,
    sanitizers=JAVASCRIPT.sanitizers,
    crypto_names=JAVASCRIPT.crypto_names,
    extra_sources_by_framework=JAVASCRIPT.extra_sources_by_framework,
)

RUBY = LanguageSecurityVocab(
    language_id="ruby",
    sources=(
        _src("rb.http", "params", "request.GET", "request.POST", "request.params"),
        _src("rb.cli", "ARGV", "gets", kind="cli"),
        _src("rb.env", "ENV", kind="env"),
    ),
    sinks=(
        _sink("rb.sql", SQL, "execute", "where", "find_by_sql"),
        _sink(
            "rb.cmd",
            CMD,
            "system",
            "exec",
            "popen",
            "Open3.capture",
            "Open3.capture2",
            "Open3.popen3",
        ),
        _sink("rb.path", PATH, "File.read", "File.open", "File.write", "IO.read", "send_file"),
        _sink(
            "rb.ssrf",
            SSRF,
            "Net::HTTP.get",
            "Net::HTTP.post",
            "Faraday.get",
            "HTTParty.get",
            "RestClient.get",
        ),
        _sink("rb.xss", XSS, "html_safe", "raw"),
        _sink("rb.deser", DESER, "Marshal.load", "YAML.load", "JSON.load"),
        _sink(
            "rb.eval",
            EVAL,
            "eval",
            "instance_eval",
            "class_eval",
            "module_eval",
            notes="Kernel#send is ordinary dynamic dispatch and is not treated as code execution.",
        ),
        _sink("rb.redir", REDIR, "redirect_to", "redirect_back"),
    ),
    sanitizers=(_san("rb.html", "ERB::Util.html_escape", "h", kind="html_encode", effective=True),),
    crypto_names=_WEAK_CRYPTO + ("Digest::MD5", "Digest::SHA1"),
    extra_sources_by_framework={
        "rails": (_src("rails.http", "params"),),
        "sinatra": (_src("sinatra.http", "params"),),
    },
)

C = LanguageSecurityVocab(
    language_id="c",
    sources=(
        _src("c.cli", "argv", kind="cli"),
        _src("c.env", "getenv", kind="env"),
        _src("c.file", "fgets", "scanf", "gets", kind="file"),
    ),
    sinks=(
        _sink(
            "c.sql",
            SQL,
            "sqlite3_exec",
            "PQexec",
            "mysql_query",
            argument_indexes=(0, 1),
        ),
        _sink("c.cmd", CMD, "system", "popen", "execl", "execv", "execvp", "execlp"),
        _sink("c.path", PATH, "open", "fopen", "openat"),
    ),
    sanitizers=(),
    crypto_names=_WEAK_CRYPTO,
)

CPP = LanguageSecurityVocab(
    language_id="cpp",
    sources=C.sources + (_src("cpp.cli", "std::cin", "std::getline", kind="cli"),),
    sinks=C.sinks
    + (_sink("cpp.cmd", CMD, "std::system"), _sink("cpp.path", PATH, "ifstream", "std::ifstream")),
    sanitizers=(),
    crypto_names=_WEAK_CRYPTO,
)

GO = LanguageSecurityVocab(
    language_id="go",
    sources=(
        _src("go.cli", "os.Args", kind="cli"),
        _src("go.env", "os.Getenv", kind="env"),
        _src("go.http", "URL.Query", "FormValue", "r.Header", "chi.URLParam", "mux.Vars"),
    ),
    sinks=(
        _sink("go.sql", SQL, "Query", "QueryRow", "Exec"),
        _sink("go.cmd", CMD, "exec.Command", "StartProcess"),
        _sink("go.path", PATH, "os.Open", "os.ReadFile", "ioutil.ReadFile", "http.ServeFile"),
        _sink("go.ssrf", SSRF, "http.Get", "http.Post", "http.NewRequest", "client.Do"),
        _sink("go.xss", XSS, "template.HTML"),
        _sink(
            "go.json",
            PDESER,
            "json.Unmarshal",
            certainty=SinkCertainty.INDICATOR,
            notes="encoding/json does not execute code; treated as a potential untrusted-input indicator.",
        ),
        _sink("go.deser", DESER, "gob.Decode"),
        _sink("go.redir", REDIR, "http.Redirect"),
    ),
    sanitizers=(
        _san(
            "go.html",
            "html.EscapeString",
            "template.HTMLEscapeString",
            kind="html_encode",
            effective=True,
        ),
    ),
    crypto_names=_WEAK_CRYPTO + ("md5.New", "sha1.New"),
)

RUST = LanguageSecurityVocab(
    language_id="rust",
    sources=(
        _src("rs.cli", "std::env::args", kind="cli"),
        _src("rs.env", "std::env::var", kind="env"),
        _src("rs.http", "req.query", "req.uri", "Path", "Query"),
    ),
    sinks=(
        _sink("rs.sql", SQL, "execute", "sqlx::query", "diesel::sql_query"),
        _sink("rs.cmd", CMD, "Command::new", "std::process::Command"),
        _sink("rs.path", PATH, "File::open", "fs::read", "fs::write"),
        _sink("rs.ssrf", SSRF, "reqwest::get", "ureq::get"),
        _sink(
            "rs.xss",
            XSS,
            "Html::from_string_unchecked",
            notes="Ordinary Html type constructors are not XSS sinks.",
        ),
        _sink(
            "rs.json",
            PDESER,
            "serde_json::from_str",
            "serde_json::from_slice",
            certainty=SinkCertainty.INDICATOR,
            notes="serde_json parsing is not code execution.",
        ),
        _sink("rs.deser", DESER, "bincode::deserialize"),
        _sink("rs.redir", REDIR, "Redirect::to"),
    ),
    sanitizers=(),
    crypto_names=_WEAK_CRYPTO,
)

JAVA = LanguageSecurityVocab(
    language_id="java",
    sources=(
        _src("java.http", "getParameter", "getHeader", "getQueryString", "request.get"),
        _src("java.env", "System.getenv", kind="env"),
        _src("java.cli", "args", kind="cli"),
    ),
    sinks=(
        _sink("java.sql", SQL, "executeQuery", "execute", "executeUpdate", "createNativeQuery"),
        _sink("java.cmd", CMD, "exec", "ProcessBuilder"),
        _sink("java.path", PATH, "File", "Files.read", "FileInputStream", "Paths.get"),
        _sink("java.ssrf", SSRF, "HttpURLConnection", "HttpClient", "URL", "RestTemplate"),
        _sink("java.xss", XSS, "getWriter", required_context="html_output"),
        _sink("java.deser", DESER, "ObjectInputStream", "readObject", "XMLDecoder", "XStream"),
        _sink("java.eval", EVAL, "ScriptEngine", "eval"),
        _sink("java.redir", REDIR, "sendRedirect", "RedirectView"),
    ),
    sanitizers=(_san("java.html", "HtmlUtils.htmlEscape", kind="html_encode", effective=True),),
    crypto_names=_WEAK_CRYPTO + ("MD5", "SHA-1"),
)

PHP = LanguageSecurityVocab(
    language_id="php",
    sources=(_src("php.http", "_GET", "_POST", "_REQUEST", "_COOKIE", "_SERVER", "_FILES"),),
    sinks=(
        _sink(
            "php.sql",
            SQL,
            "mysqli_query",
            "query",
            "exec",
            "mysql_query",
            "pg_query",
            argument_indexes=(0, 1),
        ),
        _sink("php.cmd", CMD, "system", "exec", "passthru", "shell_exec", "popen", "backtick"),
        _sink("php.path", PATH, "fopen", "file_get_contents", "include", "require"),
        _sink("php.ssrf", SSRF, "file_get_contents", "curl_exec"),
        _sink(
            "php.xss",
            XSS,
            "echo",
            "print",
            required_context="html_output",
            notes="echo/print are XSS sinks only in HTML output context.",
        ),
        _sink("php.deser", DESER, "unserialize"),
        _sink("php.eval", EVAL, "eval", "assert", "create_function"),
        _sink("php.redir", REDIR, "header"),
    ),
    sanitizers=(
        _san("php.html", "htmlspecialchars", "htmlentities", kind="html_encode", effective=True),
        _san("php.sql", "mysqli_real_escape_string", "PDO", kind="sql_parameterize"),
    ),
    crypto_names=_WEAK_CRYPTO + ("md5", "sha1"),
)

KOTLIN = LanguageSecurityVocab(
    language_id="kotlin",
    sources=(
        _src("kt.http", "queryParameters", "call.parameters", "request.headers"),
        _src("kt.env", "System.getenv", kind="env"),
        _src("kt.cli", "args", kind="cli"),
    ),
    sinks=(
        _sink("kt.sql", SQL, "executeQuery", "execute", "createNativeQuery"),
        _sink("kt.cmd", CMD, "exec", "ProcessBuilder"),
        _sink("kt.path", PATH, "File", "Files.read", "Paths.get"),
        _sink("kt.ssrf", SSRF, "URL", "HttpClient", "OkHttpClient"),
        _sink("kt.xss", XSS, "respondText"),
        _sink("kt.deser", DESER, "ObjectInputStream"),
        _sink("kt.redir", REDIR, "respondRedirect"),
    ),
    sanitizers=(),
    crypto_names=_WEAK_CRYPTO,
)

SWIFT = LanguageSecurityVocab(
    language_id="swift",
    sources=(
        _src("sw.cli", "CommandLine.arguments", kind="cli"),
        _src("sw.env", "ProcessInfo.processInfo.environment", kind="env"),
        _src("sw.http", "request.query", "URLQueryItem"),
    ),
    sinks=(
        _sink("sw.sql", SQL, "sqlite3_exec", "execute"),
        _sink("sw.cmd", CMD, "Process", "NSTask"),
        _sink("sw.path", PATH, "FileHandle", "contentsOfFile", "contentsOf"),
        _sink("sw.ssrf", SSRF, "URLSession", "URLRequest"),
        _sink("sw.deser", DESER, "NSKeyedUnarchiver"),
        _sink("sw.json", PDESER, "JSONDecoder", certainty=SinkCertainty.INDICATOR),
        _sink("sw.eval", EVAL, "NSExpression"),
    ),
    sanitizers=(),
    crypto_names=_WEAK_CRYPTO,
)

CSHARP = LanguageSecurityVocab(
    language_id="csharp",
    sources=(
        _src("cs.http", "Request.Query", "Request.Form", "Request.Headers", "HttpContext.Request"),
        _src("cs.cli", "args", kind="cli"),
        _src("cs.env", "Environment.GetEnvironmentVariable", kind="env"),
    ),
    sinks=(
        _sink("cs.sql", SQL, "ExecuteReader", "ExecuteNonQuery", "SqlCommand", "FromSqlRaw"),
        _sink("cs.cmd", CMD, "Process.Start", "ProcessStartInfo", argument_indexes=(0, 1)),
        _sink("cs.path", PATH, "File.Open", "File.ReadAllText", "File.WriteAllText", "FileStream"),
        _sink(
            "cs.ssrf",
            SSRF,
            "GetAsync",
            "GetStringAsync",
            "PostAsync",
            "SendAsync",
            "WebRequest.Create",
            "WebClient.DownloadString",
        ),
        _sink("cs.xss", XSS, "Html.Raw"),
        _sink("cs.deser", DESER, "BinaryFormatter", "SoapFormatter"),
        _sink(
            "cs.json", PDESER, "JsonConvert.DeserializeObject", certainty=SinkCertainty.INDICATOR
        ),
        _sink("cs.eval", EVAL, "CompileAssemblyFromSource"),
        _sink("cs.redir", REDIR, "Redirect", "LocalRedirect"),
    ),
    sanitizers=(
        _san("cs.html", "HtmlEncoder", "WebUtility.HtmlEncode", kind="html_encode", effective=True),
    ),
    crypto_names=_WEAK_CRYPTO + ("MD5", "SHA1"),
)

SHELL = LanguageSecurityVocab(
    language_id="shell",
    sources=(
        _src("sh.cli", r"^1$", r"^2$", r"^@$", r"^\*$", r"^1", kind="cli"),
        _src("sh.env", "ENV", kind="env"),
    ),
    sinks=(
        _sink("sh.cmd", CMD, "eval", "bash", "sh"),
        _sink("sh.eval", EVAL, "eval"),
        _sink("sh.path", PATH, "cat", "rm", "cp", "mv"),
        _sink("sh.source", CMD, "source", "."),
        _sink("sh.ssrf", SSRF, "curl", "wget"),
    ),
    sanitizers=(_san("sh.quote", "printf", kind="shell_escape"),),
    crypto_names=(),
)

HTML = LanguageSecurityVocab(
    language_id="html",
    sources=(),
    sinks=(_sink("html.script", XSS, "script", "onerror", "onload", "onclick", "javascript:"),),
    sanitizers=(),
    crypto_names=(),
)

CSS = LanguageSecurityVocab(
    language_id="css",
    sources=(),
    sinks=(_sink("css.expr", XSS, "expression", "javascript:", "behavior"),),
    sanitizers=(),
    crypto_names=(),
)

SCSS = LanguageSecurityVocab(
    language_id="scss",
    sources=(),
    sinks=CSS.sinks,
    sanitizers=(),
    crypto_names=(),
)

SQL_VOCAB = LanguageSecurityVocab(
    language_id="sql",
    sources=(),
    sinks=(
        _sink("sql.drop", SQL, "DROP", "TRUNCATE", "ALTER"),
        _sink("sql.exec", SQL, "EXECUTE", "EXEC"),
    ),
    sanitizers=(),
    crypto_names=(),
)

VOCABULARIES: dict[str, LanguageSecurityVocab] = {
    vocab.language_id: vocab
    for vocab in (
        PYTHON,
        JAVASCRIPT,
        TYPESCRIPT,
        RUBY,
        C,
        CPP,
        GO,
        RUST,
        JAVA,
        PHP,
        KOTLIN,
        SWIFT,
        CSHARP,
        SHELL,
        HTML,
        CSS,
        SCSS,
        SQL_VOCAB,
    )
}


def vocab_for(language_id: str) -> LanguageSecurityVocab | None:
    return VOCABULARIES.get(language_id.strip().lower())
