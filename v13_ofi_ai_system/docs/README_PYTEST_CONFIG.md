# Pytest 配置说明

## 问题描述

运行 pytest 时出现以下警告：

```
PytestDeprecationWarning: The configuration option "asyncio_default_fixture_loop_scope" is unset.
The event loop scope for asynchronous fixtures will default to the fixture caching scope. 
Future versions of pytest-asyncio will default the loop scope for asynchronous fixtures to function scope. 
Set the default fixture loop scope explicitly in order to avoid unexpected behavior in the future. 
Valid fixture loop scopes are: "function", "class", "module", "package", "session"
```

## 解决方案

### 方案 1：使用 pytest.ini 文件

在项目根目录创建 `pytest.ini` 文件：

```ini
[tool:pytest]
asyncio_default_fixture_loop_scope = function
asyncio_default_test_loop_scope = function
```

### 方案 2：使用 pyproject.toml 文件

在项目根目录创建或更新 `pyproject.toml` 文件：

```toml
[tool.pytest.ini_options]
asyncio_default_fixture_loop_scope = "function"
asyncio_default_test_loop_scope = "function"
testpaths = ["tests"]
python_files = ["test_*.py"]
python_classes = ["Test*"]
python_functions = ["test_*"]
addopts = "-v --tb=short"
```

### 方案 3：命令行参数

如果不想修改配置文件，可以在运行时指定参数：

```bash
pytest --asyncio-mode=strict tests/test_strategy_mode_smoke.py -v
```

## 配置说明

- **asyncio_default_fixture_loop_scope**: 异步 fixture 的事件循环作用域
- **asyncio_default_test_loop_scope**: 异步测试的事件循环作用域
- **function**: 每个测试函数使用独立的事件循环（推荐）

## 验证配置

运行以下命令验证配置是否生效：

```bash
# 运行测试，应该不再显示警告
pytest tests/test_strategy_mode_smoke.py -v

# 或者运行所有测试
pytest tests/ -v
```

## 注意事项

1. **配置文件优先级**：`pytest.ini` > `pyproject.toml` > 命令行参数
2. **作用域选择**：
   - `function`: 每个测试函数独立（推荐，最安全）
   - `class`: 每个测试类共享
   - `module`: 每个模块共享
   - `session`: 整个测试会话共享
3. **兼容性**：这些配置对同步测试没有影响

## 项目状态

✅ **pytest.ini**: 已创建
✅ **pyproject.toml**: 已创建
✅ **README.md**: 已更新配置说明
✅ **警告消除**: 配置生效后不再显示警告

现在运行 pytest 应该不会再显示 asyncio 相关的警告信息。
