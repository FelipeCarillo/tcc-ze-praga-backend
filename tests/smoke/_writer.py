
import pathlib

code = open("tests/smoke/_content.txt", encoding="utf-8").read()
pathlib.Path("tests/smoke/test_auth_users_smoke.py").write_text(code, encoding="utf-8")
print("done")
