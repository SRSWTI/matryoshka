from matryoshka.ast_extractor import extract_file


def test_extract_python_symbols_and_imports(tmp_path):
    app_dir = tmp_path / "app"
    app_dir.mkdir()
    (app_dir / "__init__.py").write_text("", encoding="utf-8")
    (app_dir / "utils.py").write_text("def verify(token: str) -> bool:\n    return True\n", encoding="utf-8")

    file_path = tmp_path / "auth.py"
    file_path.write_text(
        """
import jwt
from app.utils import verify

TOKEN_TTL = 60

@cached
def issue_token(user_id: str) -> str:
    \"\"\"Create a token\"\"\"
    return jwt.encode({"sub": user_id}, "secret")


class AuthService(BaseService):
    def validate(self, token: str) -> bool:
        return verify(token)
""".strip(),
        encoding="utf-8",
    )

    result = extract_file(file_path, repo_root=tmp_path)

    assert result.language == "python"
    assert {symbol.name for symbol in result.symbols} >= {"TOKEN_TTL", "issue_token", "AuthService", "validate"}
    assert any(edge.imported_module == "jwt" and not edge.is_internal for edge in result.import_edges)
    assert any(edge.imported_module == "app.utils" and edge.is_internal for edge in result.import_edges)
    jwt_edge = next(edge for edge in result.import_edges if edge.imported_module == "jwt")
    verify_edge = next(edge for edge in result.import_edges if edge.imported_module == "app.utils")
    function = next(symbol for symbol in result.symbols if symbol.name == "issue_token")
    assert function.decorators == ["@cached"]
    assert function.docstring == "Create a token"
    method = next(symbol for symbol in result.symbols if symbol.name == "validate")
    assert method.kind == "method"
    assert method.return_type == "bool"
    assert method.callees == ["verify"]
    assert jwt_edge.line_range is not None
    assert jwt_edge.line_range.start_line == 1
    assert verify_edge.line_range is not None
    assert verify_edge.line_range.start_line == 2
    call_site = next(site for site in result.call_sites if site.caller_name == "validate" and site.callee_name == "verify")
    assert call_site.line_range.start_line == 14


def test_extract_typescript_symbols_and_imports(tmp_path):
    file_path = tmp_path / "auth.ts"
    file_path.write_text(
        """
import jwt, { verify } from "jsonwebtoken";
import { db } from "./db";

export const TOKEN_TTL = 60;

export interface SessionToken {
  subject: string;
}

export class AuthService {
  validate(token: string): boolean {
    return verify(token) !== null;
  }
}
""".strip(),
        encoding="utf-8",
    )

    result = extract_file(file_path, repo_root=tmp_path)

    assert result.language == "typescript"
    assert {symbol.name for symbol in result.symbols} >= {"TOKEN_TTL", "SessionToken", "AuthService", "validate"}
    assert any(edge.imported_module == "jsonwebtoken" and not edge.is_internal for edge in result.import_edges)
    assert any(edge.imported_module == "./db" and edge.is_internal for edge in result.import_edges)
    method = next(symbol for symbol in result.symbols if symbol.name == "validate")
    assert method.return_type == "boolean"
    assert method.signature == "validate(token: string): boolean"
    assert method.callees == ["verify"]
    call_site = next(site for site in result.call_sites if site.caller_name == "validate" and site.callee_name == "verify")
    assert call_site.line_range.start_line == 12


def test_extract_typescript_handles_unicode_offsets_and_reexports(tmp_path):
    helper_path = tmp_path / "helpers.ts"
    helper_path.write_text(
        """
// Comment with unicode dash — exercises byte offsets.

function getProcEnv(key: string): string | undefined {
  return key;
}

export function findEnvKeys(provider: string): string[] | undefined;
export function findEnvKeys(provider: string): string[] | undefined {
  const envVars = [provider];
  return envVars.filter((envVar) => !!getProcEnv(envVar));
}
""".strip(),
        encoding="utf-8",
    )

    declaration_path = tmp_path / "bedrock-provider.d.ts"
    declaration_path.write_text('export * from "./dist/bedrock-provider.js";\n', encoding="utf-8")

    helper_result = extract_file(helper_path, repo_root=tmp_path)
    helper_symbol = next(symbol for symbol in helper_result.symbols if symbol.name == "findEnvKeys")

    assert helper_symbol.signature == "function findEnvKeys(provider: string): string[] | undefined"
    assert helper_symbol.callees == ["filter", "getProcEnv"]

    declaration_result = extract_file(declaration_path, repo_root=tmp_path)
    assert any(
        edge.imported_module == "./dist/bedrock-provider.js" and edge.is_internal and edge.names == ["*"]
        for edge in declaration_result.import_edges
    )
    export_edge = next(edge for edge in declaration_result.import_edges if edge.imported_module == "./dist/bedrock-provider.js")
    assert export_edge.line_range is not None
    assert export_edge.line_range.start_line == 1


def test_extract_typescript_compacts_variable_function_signatures(tmp_path):
    file_path = tmp_path / "providers.ts"
    file_path.write_text(
        """
export const streamBedrock = (
  model: string,
  context: string,
) => {
  const output = { role: "assistant" };
  return `${model}:${context}:${output.role}`;
};
""".strip(),
        encoding="utf-8",
    )

    result = extract_file(file_path, repo_root=tmp_path)

    symbol = next(symbol for symbol in result.symbols if symbol.name == "streamBedrock")
    assert symbol.signature == 'streamBedrock = ( model: string, context: string, ) => { ... }'