import pytest
import sys

if __name__ == "__main__":
    exit_code = pytest.main(["-v", "-p", "no:warnings", "tests/test_main.py"])
    
    if exit_code == 0:
        print("\n Все тесты пройдены успешно.")
    else:
        print("\n Обнаружены ошибки в модулях.")
    
    sys.exit(exit_code)