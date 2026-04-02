# OCR Finder

本地离线 OCR 图片搜索工具，递归扫描文件夹，查找包含目标文本的图片。

## 功能特点

- **本地离线运行** - 使用 RapidOCR ONNX Runtime，无需联网
- **中英界面切换** - 支持 GUI 界面中英文切换
- **多关键词搜索** - 支持多个关键词，使用逗号、分号或竖线分隔
- **多种匹配模式** - AND（全部匹配）、OR（任一匹配）、NOT（都不匹配）
- **进程优先级控制** - 低/中/高三档
- **多种输出方式** - 硬链接（默认）、软链接、复制
- **自动去重** - 基于文件内容哈希

## 支持的图片格式

`.jpg`, `.jpeg`, `.png`, `.bmp`, `.gif`, `.tiff`, `.tif`, `.webp`

## 环境要求

- Python 3.10+
- Windows

## 快速开始

### 1. 设置环境

```bat
setup_cpu_venv.bat
```

### 2. 运行 GUI

```bash
venv\Scripts\activate
python ocr_finder_gui.py
```

### 3. 或使用 CLI

```bash
venv\Scripts\activate
python ocr_finder.py -t "你好" -s ./images
```

## 使用说明

### GUI 操作

1. 选择界面语言（中文/English）
2. 输入目标关键词（多个关键词用逗号分隔）
3. 选择源文件夹和输出文件夹
4. 选择匹配模式、进程优先级、输出方式
5. 点击"开始"开始扫描

### CLI 参数

| 参数 | 说明 | 默认值 |
|---|---|---|
| `-t`, `--target` | 目标关键词（多个用逗号分隔）| 必填 |
| `-s`, `--source` | 源目录 | 必填 |
| `-o`, `--output` | 输出目录 | `./output` |
| `-m`, `--mode` | 匹配模式：`and`/`or`/`not` | `or` |
| `--output-mode` | 输出方式：`hardlink`/`symlink`/`copy` | `hardlink` |
| `-c`, `--cpu` | CPU 限制百分比 (1-100) | `50` |
| `-v`, `--verbose` | 详细日志 | `False` |

### 示例

```bash
# 搜索单个关键词
python ocr_finder.py -t "你好" -s ./images

# 搜索多个关键词（任一匹配）
python ocr_finder.py -t "hello,world,test" -s ./docs -m or

# 搜索多个关键词（全部匹配）
python ocr_finder.py -t "A,B,C" -s ./photos -m and

# 排除包含特定关键词的图片
python ocr_finder.py -t "广告,推广" -s ./archive -m not
```

## 打包离线 EXE

```bat
build_cpu.bat
```

输出位于 `release\ocr_finder_gui.dist\OCRFinder.exe`

## 项目结构

```text
OCRfinder/
├── ocr_finder.py          # 核心逻辑与 CLI
├── ocr_finder_gui.py      # Tkinter GUI
├── setup_cpu_venv.bat     # 环境设置
├── build_cpu.bat          # 打包脚本
├── requirements.txt       # Python 依赖
├── SPEC.md                # 规格说明
└── README.md              # 使用说明
```

## 技术栈

- RapidOCR (ONNX Runtime) - 轻量级 CPU OCR
- psutil - 进程管理
- Tkinter - GUI
- Nuitka - 打包工具

## 许可证

MIT License