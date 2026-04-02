import io
import os
import queue
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import chinese_ocr_finder as module
import chinese_ocr_finder_gui as gui_module


class OCRFinderTests(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tempdir.name)

    def tearDown(self):
        self.tempdir.cleanup()

    def make_file(self, relative_path: str, content: bytes = b"data") -> Path:
        path = self.root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        return path

    def make_finder(
        self,
        target: str = "你好",
        source: Path | None = None,
        output: Path | None = None,
        **kwargs,
    ):
        source = source or self.root / "source"
        output = output or self.root / "output"
        source.mkdir(parents=True, exist_ok=True)
        output.mkdir(parents=True, exist_ok=True)
        return module.OCRFinder(target, str(source), str(output), **kwargs)

    def test_normalize_build_profile_returns_supported_or_default_values(self):
        self.assertEqual(module.normalize_build_profile("cpu"), "cpu")
        self.assertEqual(module.normalize_build_profile(" GPU "), "gpu")
        self.assertEqual(module.normalize_build_profile("unknown"), module.DEFAULT_BUILD_PROFILE)
        self.assertEqual(module.normalize_build_profile(None), module.DEFAULT_BUILD_PROFILE)

    def test_normalize_ocr_backend_returns_supported_or_default_values(self):
        self.assertEqual(module.normalize_ocr_backend("paddle"), "paddle")
        self.assertEqual(module.normalize_ocr_backend(" RapidOCR "), "rapidocr")
        self.assertEqual(module.normalize_ocr_backend("unknown"), module.DEFAULT_OCR_BACKEND)
        self.assertEqual(module.normalize_ocr_backend(None), module.DEFAULT_OCR_BACKEND)

    def test_runtime_defaults_can_be_read_from_environment(self):
        with patch.dict(os.environ, {
            "OCR_FINDER_BUILD_PROFILE": "cpu",
            "OCR_FINDER_OCR_BACKEND": "rapidocr",
        }, clear=False):
            self.assertEqual(module.get_default_build_profile(), "cpu")
            self.assertEqual(module.get_default_ocr_backend(), "rapidocr")

    def test_finder_uses_environment_runtime_defaults_when_not_provided(self):
        with patch.dict(os.environ, {
            "OCR_FINDER_BUILD_PROFILE": "cpu",
            "OCR_FINDER_OCR_BACKEND": "rapidocr",
        }, clear=False):
            finder = self.make_finder()

        self.assertEqual(finder.build_profile, "cpu")
        self.assertEqual(finder.backend, "rapidocr")

    def test_get_image_files_skips_output_directory_inside_source(self):
        source = self.root / "source"
        output = source / "output"
        self.make_file("source/keep.png")
        self.make_file("source/nested/keep.jpg")
        self.make_file("source/output/skip.png")
        self.make_file("source/output/nested/skip2.webp")
        self.make_file("source/readme.txt")

        finder = self.make_finder(source=source, output=output)

        image_names = {path.relative_to(source).as_posix() for path in finder.get_image_files(source)}

        self.assertEqual(image_names, {"keep.png", "nested/keep.jpg"})

    def test_get_input_files_includes_images_and_pdfs_only(self):
        source = self.root / "source"
        output = self.root / "output"
        self.make_file("source/keep.png")
        self.make_file("source/report.pdf")
        self.make_file("source/notes.txt", b"keyword")
        self.make_file("source/nested/readme.md", b"keyword")
        self.make_file("source/ignore.csv")

        finder = self.make_finder(source=source, output=output)
        input_names = {path.relative_to(source).as_posix() for path in finder.get_input_files(source)}

        self.assertEqual(input_names, {"keep.png", "report.pdf"})

    def test_extract_text_for_file_routes_images_and_pdfs(self):
        finder = self.make_finder()
        pdf_path = self.root / "source" / "report.pdf"
        image_path = self.root / "source" / "image.png"

        with patch.object(finder, "perform_pdf_ocr", return_value="pdf-result") as pdf_mock, \
             patch.object(finder, "perform_ocr", return_value="image-result") as ocr_mock:
            self.assertEqual(finder.extract_text_for_file(pdf_path), "pdf-result")
            self.assertEqual(finder.extract_text_for_file(image_path), "image-result")

        pdf_mock.assert_called_once_with(pdf_path)
        ocr_mock.assert_called_once_with(image_path)

    def test_perform_pdf_ocr_stops_after_target_match(self):
        finder = self.make_finder(target="你")
        pdf_path = self.root / "source" / "report.pdf"

        with patch.object(finder, "iter_pdf_page_images", return_value=[(1, "page-1"), (2, "page-2"), (3, "page-3")]), \
             patch.object(finder, "perform_ocr", side_effect=["第一页", "这里有你", "不会执行"]) as ocr_mock:
            self.assertEqual(finder.perform_pdf_ocr(pdf_path), "第一页\n这里有你")

        self.assertEqual(ocr_mock.call_count, 2)

    def test_contains_target_char_uses_normalized_substring_matching(self):
        finder = self.make_finder(target="你好")

        self.assertTrue(finder.contains_target_char("前缀你好后缀"))
        self.assertFalse(finder.contains_target_char("今天真好"))
        self.assertFalse(finder.contains_target_char("你在哪里"))

    def test_contains_target_text_uses_character_hit_for_single_character_targets(self):
        finder = self.make_finder(target="你")

        self.assertTrue(finder.contains_target_text("这里有你"))
        self.assertTrue(finder.contains_target_text("你好"))
        self.assertFalse(finder.contains_target_text("世界和平"))

    def test_perform_ocr_extracts_text_from_supported_result_shapes(self):
        finder = self.make_finder()
        fake_image = MagicMock()
        image_context = MagicMock()
        image_context.__enter__.return_value = fake_image
        mock_image_module = MagicMock()
        mock_image_module.open.return_value = image_context

        with patch.object(module, "Image", mock_image_module):
            finder.ocr = Mock()
            finder.ocr.predict.return_value = [
                {"rec_text": "你好"},
                {"rec_texts": ["世界", ""]},
            ]
            self.assertEqual(finder.perform_ocr(Path("dict.png")), "你好\n世界")

            finder.ocr.predict.return_value = [
                ([(0, 0)], ("旧格式", 0.99)),
                ([(1, 1)], None),
            ]
            self.assertEqual(finder.perform_ocr(Path("legacy.png")), "旧格式")

        self.assertEqual(fake_image.verify.call_count, 2)

    def test_run_ocr_backend_prefers_active_backend(self):
        finder = self.make_finder()
        finder.ocr_backend = Mock()
        finder.ocr_backend.predict.return_value = "backend-result"
        finder.ocr = Mock()
        finder.ocr.predict.return_value = "legacy-result"

        self.assertEqual(finder._run_ocr_backend("input"), "backend-result")
        finder.ocr_backend.predict.assert_called_once_with("input")
        finder.ocr.predict.assert_not_called()

    def test_run_ocr_backend_falls_back_to_legacy_ocr_attribute(self):
        finder = self.make_finder()
        finder.ocr = Mock()
        finder.ocr.predict.return_value = "legacy-result"

        self.assertEqual(finder._run_ocr_backend("input"), "legacy-result")
        finder.ocr.predict.assert_called_once_with("input")

    def test_copy_to_output_skips_existing_duplicate_content(self):
        source = self.make_file("source/fixture.png", b"same-content")
        output = self.root / "output"
        self.make_file("output/already-there.png", b"same-content")
        finder = self.make_finder(source=self.root / "source", output=output)

        result = finder.copy_to_output(source)

        self.assertEqual(result, "duplicate")
        self.assertEqual(sorted(path.name for path in output.iterdir()), ["already-there.png"])

    def test_copy_to_output_renames_when_name_conflicts_with_different_content(self):
        source = self.make_file("source/fixture.png", b"new-content")
        output = self.root / "output"
        self.make_file("output/fixture.png", b"old-content")
        finder = self.make_finder(source=self.root / "source", output=output)

        result = finder.copy_to_output(source)

        self.assertEqual(result, "copied")
        self.assertTrue((output / "fixture_1.png").exists())
        self.assertEqual((output / "fixture_1.png").read_bytes(), b"new-content")

    def test_get_runtime_resource_dir_uses_meipass_for_frozen_runtime(self):
        meipass_path = self.root / "release" / "runtime"

        with patch.object(module.sys, "frozen", True, create=True), \
             patch.object(module.sys, "_MEIPASS", str(meipass_path), create=True), \
             patch.object(module.sys, "executable", str(self.root / "release" / "OCRFinder.exe")):
            self.assertEqual(module.get_runtime_resource_dir(), meipass_path)

    def test_get_runtime_resource_dir_uses_executable_parent_for_frozen_runtime_without_meipass(self):
        executable_path = self.root / "release" / "OCRFinder.exe"

        with patch.object(module.sys, "frozen", True, create=True), \
             patch.object(module.sys, "_MEIPASS", None, create=True), \
             patch.object(module.sys, "executable", str(executable_path)):
            self.assertEqual(module.get_runtime_resource_dir(), executable_path.parent)

    def test_get_runtime_resource_dir_falls_back_to_module_parent_when_frozen_runtime_has_no_paths(self):
        module_file = self.root / "source" / "chinese_ocr_finder.py"
        original_value = module.__dict__.get("__compiled__", None)
        had_original = "__compiled__" in module.__dict__

        try:
            module.__dict__.pop("__compiled__", None)
            with patch.object(module.sys, "frozen", True, create=True), \
                 patch.object(module.sys, "_MEIPASS", None, create=True), \
                 patch.object(module.sys, "executable", None), \
                 patch.object(module, "__file__", str(module_file)):
                self.assertEqual(module.get_runtime_resource_dir(), module_file.parent)
        finally:
            if had_original:
                module.__dict__["__compiled__"] = original_value
            else:
                module.__dict__.pop("__compiled__", None)

    def test_get_runtime_resource_dir_uses_argv_parent_for_compiled_runtime(self):
        executable_path = self.root / "release" / "compiled" / "OCRFinder.exe"
        original_value = module.__dict__.get("__compiled__", None)
        had_original = "__compiled__" in module.__dict__

        try:
            module.__dict__["__compiled__"] = object()
            with patch.object(module.sys, "frozen", False, create=True), \
                 patch.object(module.sys, "argv", [str(executable_path)]), \
                 patch.object(module.sys, "executable", str(self.root / "ignored" / "python.exe")):
                self.assertEqual(module.get_runtime_resource_dir(), executable_path.parent)
        finally:
            if had_original:
                module.__dict__["__compiled__"] = original_value
            else:
                module.__dict__.pop("__compiled__", None)

    def test_get_runtime_resource_dir_falls_back_to_executable_for_compiled_runtime(self):
        executable_path = self.root / "release" / "fallback" / "OCRFinder.exe"
        original_value = module.__dict__.get("__compiled__", None)
        had_original = "__compiled__" in module.__dict__

        try:
            module.__dict__["__compiled__"] = object()
            with patch.object(module.sys, "frozen", False, create=True), \
                 patch.object(module.sys, "argv", [None]), \
                 patch.object(module.sys, "executable", str(executable_path)):
                self.assertEqual(module.get_runtime_resource_dir(), executable_path.parent)
        finally:
            if had_original:
                module.__dict__["__compiled__"] = original_value
            else:
                module.__dict__.pop("__compiled__", None)

    def test_get_runtime_resource_dir_falls_back_to_module_parent_when_compiled_runtime_has_no_paths(self):
        module_file = self.root / "source" / "chinese_ocr_finder.py"
        original_value = module.__dict__.get("__compiled__", None)
        had_original = "__compiled__" in module.__dict__

        try:
            module.__dict__["__compiled__"] = object()
            with patch.object(module.sys, "frozen", False, create=True), \
                 patch.object(module.sys, "argv", [None]), \
                 patch.object(module.sys, "executable", None), \
                 patch.object(module, "__file__", str(module_file)):
                self.assertEqual(module.get_runtime_resource_dir(), module_file.parent)
        finally:
            if had_original:
                module.__dict__["__compiled__"] = original_value
            else:
                module.__dict__.pop("__compiled__", None)

    def test_get_runtime_resource_dir_uses_module_parent_for_source_runtime(self):
        module_file = self.root / "source" / "chinese_ocr_finder.py"
        original_value = module.__dict__.get("__compiled__", None)
        had_original = "__compiled__" in module.__dict__

        try:
            module.__dict__.pop("__compiled__", None)
            with patch.object(module.sys, "frozen", False, create=True), \
                 patch.object(module, "__file__", str(module_file)):
                self.assertEqual(module.get_runtime_resource_dir(), module_file.parent)
        finally:
            if had_original:
                module.__dict__["__compiled__"] = original_value
            else:
                module.__dict__.pop("__compiled__", None)

    def test_is_packaged_runtime_detects_frozen_runtime_without_compiled_marker(self):
        original_value = module.__dict__.get("__compiled__", None)
        had_original = "__compiled__" in module.__dict__

        try:
            module.__dict__.pop("__compiled__", None)
            with patch.object(module.sys, "frozen", True, create=True):
                self.assertTrue(module.is_packaged_runtime())
        finally:
            if had_original:
                module.__dict__["__compiled__"] = original_value
            else:
                module.__dict__.pop("__compiled__", None)

    def test_is_packaged_runtime_detects_compiled_runtime_without_sys_frozen(self):
        original_value = module.__dict__.get("__compiled__", None)
        had_original = "__compiled__" in module.__dict__

        try:
            module.__dict__["__compiled__"] = object()
            with patch.object(module.sys, "frozen", False, create=True):
                self.assertTrue(module.is_packaged_runtime())
        finally:
            if had_original:
                module.__dict__["__compiled__"] = original_value
            else:
                module.__dict__.pop("__compiled__", None)

    def test_is_packaged_runtime_returns_false_for_source_runtime(self):
        original_value = module.__dict__.get("__compiled__", None)
        had_original = "__compiled__" in module.__dict__

        try:
            module.__dict__.pop("__compiled__", None)
            with patch.object(module.sys, "frozen", False, create=True):
                self.assertFalse(module.is_packaged_runtime())
        finally:
            if had_original:
                module.__dict__["__compiled__"] = original_value
            else:
                module.__dict__.pop("__compiled__", None)

    def test_get_module_spec_returns_find_spec_result(self):
        found_spec = object()

        with patch.object(module.importlib.util, "find_spec", return_value=found_spec) as find_spec_mock:
            self.assertIs(module._get_module_spec("paddlex"), found_spec)

        find_spec_mock.assert_called_once_with("paddlex")

    def test_get_module_spec_raises_import_error_when_spec_is_missing(self):
        with patch.object(module.importlib.util, "find_spec", return_value=None) as find_spec_mock:
            with self.assertRaisesRegex(ImportError, "Unable to locate import spec for paddlex"):
                module._get_module_spec("paddlex")

        find_spec_mock.assert_called_once_with("paddlex")

    def test_get_package_root_prefers_existing_origin_parent_when_available(self):
        package_init = self.root / "site-packages" / "paddlex" / "__init__.py"
        package_init.parent.mkdir(parents=True, exist_ok=True)
        package_init.write_text("# paddlex\n", encoding="utf-8")
        package_spec = type(
            "Spec",
            (),
            {"origin": str(package_init), "submodule_search_locations": [str(self.root / "ignored")]},
        )()

        with patch.object(module, "_get_module_spec", return_value=package_spec):
            self.assertEqual(module._get_package_root("paddlex"), package_init.parent.resolve())

    def test_get_package_root_uses_search_locations_when_origin_is_frozen(self):
        package_root = self.root / "site-packages" / "paddlex"
        package_root.mkdir(parents=True, exist_ok=True)
        package_spec = type(
            "Spec",
            (),
            {"origin": "frozen", "submodule_search_locations": [str(package_root)]},
        )()

        with patch.object(module, "_get_module_spec", return_value=package_spec):
            self.assertEqual(module._get_package_root("paddlex"), package_root.resolve())

    def test_get_package_root_uses_search_locations_when_origin_is_built_in(self):
        package_root = self.root / "site-packages" / "paddlex"
        package_root.mkdir(parents=True, exist_ok=True)
        package_spec = type(
            "Spec",
            (),
            {"origin": "built-in", "submodule_search_locations": [str(package_root)]},
        )()

        with patch.object(module, "_get_module_spec", return_value=package_spec):
            self.assertEqual(module._get_package_root("paddlex"), package_root.resolve())

    def test_get_package_root_uses_search_locations_when_origin_path_is_missing(self):
        package_root = self.root / "site-packages" / "paddlex"
        package_root.mkdir(parents=True, exist_ok=True)
        missing_init = self.root / "missing" / "paddlex" / "__init__.py"
        package_spec = type(
            "Spec",
            (),
            {"origin": str(missing_init), "submodule_search_locations": [str(package_root)]},
        )()

        with patch.object(module, "_get_module_spec", return_value=package_spec):
            self.assertEqual(module._get_package_root("paddlex"), package_root.resolve())

    def test_get_package_root_returns_none_when_origin_and_search_locations_are_missing(self):
        missing_root = self.root / "site-packages" / "missing-paddlex"
        package_spec = type(
            "Spec",
            (),
            {"origin": str(missing_root / "__init__.py"), "submodule_search_locations": [str(missing_root)]},
        )()

        with patch.object(module, "_get_module_spec", return_value=package_spec):
            self.assertIsNone(module._get_package_root("paddlex"))

    def test_get_package_root_skips_missing_search_locations_until_existing_path(self):
        missing_root = self.root / "site-packages" / "missing-paddlex"
        package_root = self.root / "site-packages" / "paddlex"
        package_root.mkdir(parents=True, exist_ok=True)
        package_spec = type(
            "Spec",
            (),
            {"origin": None, "submodule_search_locations": [str(missing_root), str(package_root)]},
        )()

        with patch.object(module, "_get_module_spec", return_value=package_spec):
            self.assertEqual(module._get_package_root("paddlex"), package_root.resolve())

    def test_ensure_virtual_package_repairs_existing_packaged_module_search_locations(self):
        package_name = "virtualpkg"
        runtime_root = self.root / "release"
        existing_module = module.types.ModuleType(package_name)
        existing_module.__spec__ = module.importlib.machinery.ModuleSpec("other.module", loader=object())
        packaged_spec = type(
            "Spec",
            (),
            {"origin": "frozen", "submodule_search_locations": None},
        )()

        with patch.dict(module.sys.modules, {package_name: existing_module}, clear=False), \
             patch.object(module, "_get_module_spec", return_value=packaged_spec) as get_module_spec_mock, \
             patch.object(module, "is_packaged_runtime", return_value=True), \
             patch.object(module, "get_runtime_resource_dir", return_value=runtime_root):
            result = module._ensure_virtual_package(package_name)

        self.assertIs(result, existing_module)
        self.assertEqual(existing_module.__file__, "frozen")
        self.assertEqual(existing_module.__path__, [str(runtime_root / package_name)])
        self.assertEqual(existing_module.__spec__.name, package_name)
        self.assertEqual(existing_module.__spec__.submodule_search_locations, [str(runtime_root / package_name)])
        get_module_spec_mock.assert_called_once_with(package_name)

    def test_ensure_virtual_package_creates_top_level_virtual_package(self):
        package_name = "virtualpkg"
        package_root = self.root / "site-packages" / package_name
        package_spec = type(
            "Spec",
            (),
            {"origin": str(package_root / "__init__.py"), "submodule_search_locations": [str(package_root)]},
        )()

        with patch.dict(module.sys.modules, {package_name: None}, clear=False), \
             patch.object(module, "_get_module_spec", return_value=package_spec) as get_module_spec_mock:
            result = module._ensure_virtual_package(package_name)
            installed_module = module.sys.modules[package_name]

        self.assertIs(result, installed_module)
        self.assertEqual(installed_module.__file__, str(package_root / "__init__.py"))
        self.assertEqual(installed_module.__package__, package_name)
        self.assertEqual(installed_module.__path__, [str(package_root)])
        self.assertEqual(installed_module.__spec__.submodule_search_locations, [str(package_root)])
        get_module_spec_mock.assert_called_once_with(package_name)

    def test_ensure_virtual_package_creates_nested_virtual_package_and_links_parent(self):
        parent_name = "virtualpkg"
        package_name = f"{parent_name}.subpkg"
        parent_root = self.root / "site-packages" / parent_name
        child_root = parent_root / "subpkg"
        parent_spec = type(
            "Spec",
            (),
            {"origin": str(parent_root / "__init__.py"), "submodule_search_locations": [str(parent_root)]},
        )()
        child_spec = type(
            "Spec",
            (),
            {"origin": str(child_root / "__init__.py"), "submodule_search_locations": [str(child_root)]},
        )()
        spec_map = {parent_name: parent_spec, package_name: child_spec}

        with patch.dict(module.sys.modules, {parent_name: None, package_name: None}, clear=False), \
             patch.object(module, "_get_module_spec", side_effect=lambda name: spec_map[name]) as get_module_spec_mock:
            result = module._ensure_virtual_package(package_name)
            parent_module = module.sys.modules[parent_name]
            installed_module = module.sys.modules[package_name]

        self.assertIs(result, installed_module)
        self.assertEqual(installed_module.__file__, str(child_root / "__init__.py"))
        self.assertEqual(installed_module.__package__, package_name)
        self.assertEqual(installed_module.__path__, [str(child_root)])
        self.assertEqual(installed_module.__spec__.submodule_search_locations, [str(child_root)])
        self.assertIs(parent_module.subpkg, installed_module)
        self.assertEqual([call.args[0] for call in get_module_spec_mock.call_args_list], [parent_name, package_name])

    def test_ensure_virtual_package_uses_package_init_origin_when_search_locations_missing(self):
        package_name = "virtualpkg"
        package_root = self.root / "site-packages" / package_name
        package_init = package_root / "__init__.py"
        package_root.mkdir(parents=True, exist_ok=True)
        package_init.write_text("", encoding="utf-8")
        package_spec = type(
            "Spec",
            (),
            {"origin": str(package_init), "submodule_search_locations": None},
        )()

        with patch.dict(module.sys.modules, {package_name: None}, clear=False), \
             patch.object(module, "_get_module_spec", return_value=package_spec) as get_module_spec_mock:
            result = module._ensure_virtual_package(package_name)
            installed_module = module.sys.modules[package_name]

        self.assertIs(result, installed_module)
        self.assertEqual(installed_module.__file__, str(package_init))
        self.assertEqual(installed_module.__package__, package_name)
        self.assertEqual(installed_module.__path__, [str(package_root)])
        self.assertEqual(installed_module.__spec__.submodule_search_locations, [str(package_root)])
        get_module_spec_mock.assert_called_once_with(package_name)

    def test_ensure_virtual_package_uses_packaged_runtime_fallback_when_frozen_spec_has_no_search_locations(self):
        parent_name = "virtualpkg"
        package_name = f"{parent_name}.subpkg"
        runtime_root = self.root / "release"
        packaged_spec = type(
            "Spec",
            (),
            {"origin": "frozen", "submodule_search_locations": None},
        )()
        spec_map = {parent_name: packaged_spec, package_name: packaged_spec}

        with patch.dict(module.sys.modules, {parent_name: None, package_name: None}, clear=False), \
             patch.object(module, "_get_module_spec", side_effect=lambda name: spec_map[name]) as get_module_spec_mock, \
             patch.object(module, "is_packaged_runtime", return_value=True), \
             patch.object(module, "get_runtime_resource_dir", return_value=runtime_root):
            result = module._ensure_virtual_package(package_name)
            parent_module = module.sys.modules[parent_name]
            installed_module = module.sys.modules[package_name]

        self.assertIs(result, installed_module)
        self.assertEqual(parent_module.__path__, [str(runtime_root / parent_name)])
        self.assertEqual(installed_module.__path__, [str(runtime_root / parent_name / "subpkg")])
        self.assertEqual(
            installed_module.__spec__.submodule_search_locations,
            [str(runtime_root / parent_name / "subpkg")],
        )
        self.assertIs(parent_module.subpkg, installed_module)
        self.assertEqual([call.args[0] for call in get_module_spec_mock.call_args_list], [parent_name, package_name])

    def test_ensure_virtual_package_raises_import_error_for_non_package_spec(self):
        package_name = "virtualpkg"
        package_spec = type(
            "Spec",
            (),
            {"origin": str(self.root / "site-packages" / "virtualpkg.py"), "submodule_search_locations": None},
        )()

        with patch.dict(module.sys.modules, {package_name: None}, clear=False), \
             patch.object(module, "_get_module_spec", return_value=package_spec):
            with self.assertRaisesRegex(ImportError, f"{package_name} is not a package"):
                module._ensure_virtual_package(package_name)
            self.assertIsNone(module.sys.modules[package_name])

    def test_load_module_without_package_init_returns_existing_sys_modules_entry(self):
        module_name = "testpkg.child"
        existing_module = module.types.ModuleType(module_name)

        with patch.dict(module.sys.modules, {module_name: existing_module}, clear=False), \
             patch.object(module, "_ensure_virtual_package") as ensure_package_mock, \
             patch.object(module, "_get_module_spec") as get_module_spec_mock, \
             patch.object(module.importlib, "import_module") as import_module_mock:
            result = module._load_module_without_package_init(module_name)

        self.assertIs(result, existing_module)
        ensure_package_mock.assert_not_called()
        get_module_spec_mock.assert_not_called()
        import_module_mock.assert_not_called()

    def test_load_module_without_package_init_loads_top_level_module_from_file_spec(self):
        module_name = "top_level_module"
        module_path = self.root / "site-packages" / f"{module_name}.py"
        module_path.parent.mkdir(parents=True, exist_ok=True)
        module_path.write_text("LOADED_VALUE = 789\n", encoding="utf-8")
        file_spec = type(
            "Spec",
            (),
            {"origin": str(module_path), "submodule_search_locations": None},
        )()

        with patch.dict(module.sys.modules, {module_name: None}, clear=False), \
             patch.object(module, "_ensure_virtual_package") as ensure_package_mock, \
             patch.object(module, "_get_module_spec", return_value=file_spec), \
             patch.object(module.importlib, "import_module") as import_module_mock:
            result = module._load_module_without_package_init(module_name)
            self.assertIs(module.sys.modules[module_name], result)

        self.assertEqual(result.LOADED_VALUE, 789)
        ensure_package_mock.assert_not_called()
        import_module_mock.assert_not_called()

    def test_load_module_without_package_init_falls_back_to_import_module_for_frozen_spec(self):
        parent_name = "testpkg"
        module_name = f"{parent_name}.child"
        parent_module = module.types.ModuleType(parent_name)
        imported_module = object()
        frozen_spec = type(
            "Spec",
            (),
            {"origin": "frozen", "submodule_search_locations": None},
        )()

        with patch.dict(module.sys.modules, {parent_name: parent_module}, clear=False), \
             patch.object(module, "_ensure_virtual_package", return_value=parent_module) as ensure_package_mock, \
             patch.object(module, "_get_module_spec", return_value=frozen_spec), \
             patch.object(module.importlib, "import_module", return_value=imported_module) as import_module_mock:
            result = module._load_module_without_package_init(module_name)

        self.assertIs(result, imported_module)
        self.assertIs(parent_module.child, imported_module)
        ensure_package_mock.assert_called_once_with(parent_name)
        import_module_mock.assert_called_once_with(module_name)

    def test_load_module_without_package_init_falls_back_to_import_module_for_built_in_spec(self):
        parent_name = "testpkg"
        module_name = f"{parent_name}.child"
        parent_module = module.types.ModuleType(parent_name)
        imported_module = object()
        built_in_spec = type(
            "Spec",
            (),
            {"origin": "built-in", "submodule_search_locations": None},
        )()

        with patch.dict(module.sys.modules, {parent_name: parent_module}, clear=False), \
             patch.object(module, "_ensure_virtual_package", return_value=parent_module) as ensure_package_mock, \
             patch.object(module, "_get_module_spec", return_value=built_in_spec), \
             patch.object(module.importlib.util, "spec_from_file_location") as spec_from_file_location_mock, \
             patch.object(module.importlib, "import_module", return_value=imported_module) as import_module_mock:
            result = module._load_module_without_package_init(module_name)

        self.assertIs(result, imported_module)
        self.assertIs(parent_module.child, imported_module)
        ensure_package_mock.assert_called_once_with(parent_name)
        spec_from_file_location_mock.assert_not_called()
        import_module_mock.assert_called_once_with(module_name)

    def test_load_module_without_package_init_falls_back_to_import_module_when_spec_has_no_origin(self):
        parent_name = "testpkg"
        module_name = f"{parent_name}.child"
        parent_module = module.types.ModuleType(parent_name)
        imported_module = object()
        module_spec = type(
            "Spec",
            (),
            {"origin": None, "submodule_search_locations": None},
        )()

        with patch.dict(module.sys.modules, {parent_name: parent_module}, clear=False), \
             patch.object(module, "_ensure_virtual_package", return_value=parent_module) as ensure_package_mock, \
             patch.object(module, "_get_module_spec", return_value=module_spec), \
             patch.object(module.importlib.util, "spec_from_file_location") as spec_from_file_location_mock, \
             patch.object(module.importlib, "import_module", return_value=imported_module) as import_module_mock:
            result = module._load_module_without_package_init(module_name)

        self.assertIs(result, imported_module)
        self.assertIs(parent_module.child, imported_module)
        ensure_package_mock.assert_called_once_with(parent_name)
        spec_from_file_location_mock.assert_not_called()
        import_module_mock.assert_called_once_with(module_name)

    def test_load_module_without_package_init_loads_top_level_module_from_file_spec(self):
        module_name = "standalone"
        module_path = self.root / "site-packages" / f"{module_name}.py"
        module_path.parent.mkdir(parents=True, exist_ok=True)
        module_path.write_text("LOADED_VALUE = 789\n", encoding="utf-8")
        file_spec = type(
            "Spec",
            (),
            {"origin": str(module_path), "submodule_search_locations": None},
        )()

        with patch.object(module, "_ensure_virtual_package") as ensure_package_mock, \
             patch.object(module, "_get_module_spec", return_value=file_spec), \
             patch.object(module.importlib.util, "spec_from_file_location", wraps=module.importlib.util.spec_from_file_location) as spec_from_file_location_mock, \
             patch.object(module.importlib, "import_module") as import_module_mock:
            result = module._load_module_without_package_init(module_name)
            self.assertIs(module.sys.modules[module_name], result)

        self.assertEqual(result.LOADED_VALUE, 789)
        ensure_package_mock.assert_not_called()
        spec_from_file_location_mock.assert_called_once_with(module_name, module_path)
        import_module_mock.assert_not_called()

    def test_load_module_without_package_init_loads_module_from_file_spec(self):
        parent_name = "testpkg"
        module_name = f"{parent_name}.child"
        parent_module = module.types.ModuleType(parent_name)
        module_path = self.root / "site-packages" / parent_name / "child.py"
        module_path.parent.mkdir(parents=True, exist_ok=True)
        module_path.write_text("LOADED_VALUE = 123\n", encoding="utf-8")
        file_spec = type(
            "Spec",
            (),
            {"origin": str(module_path), "submodule_search_locations": None},
        )()

        with patch.dict(module.sys.modules, {parent_name: parent_module}, clear=False), \
             patch.object(module, "_ensure_virtual_package", return_value=parent_module) as ensure_package_mock, \
             patch.object(module, "_get_module_spec", return_value=file_spec), \
             patch.object(module.importlib, "import_module") as import_module_mock:
            result = module._load_module_without_package_init(module_name)
            self.assertIs(module.sys.modules[module_name], result)

        self.assertEqual(result.LOADED_VALUE, 123)
        self.assertIs(parent_module.child, result)
        ensure_package_mock.assert_called_once_with(parent_name)
        import_module_mock.assert_not_called()

    def test_load_module_without_package_init_preserves_package_search_locations_for_file_spec(self):
        parent_name = "testpkg"
        module_name = f"{parent_name}.childpkg"
        parent_module = module.types.ModuleType(parent_name)
        package_root = self.root / "site-packages" / parent_name / "childpkg"
        package_init = package_root / "__init__.py"
        package_root.mkdir(parents=True, exist_ok=True)
        package_init.write_text("LOADED_VALUE = 456\n", encoding="utf-8")
        package_spec = type(
            "Spec",
            (),
            {"origin": str(package_init), "submodule_search_locations": [str(package_root)]},
        )()

        with patch.dict(module.sys.modules, {parent_name: parent_module}, clear=False), \
             patch.object(module, "_ensure_virtual_package", return_value=parent_module) as ensure_package_mock, \
             patch.object(module, "_get_module_spec", return_value=package_spec), \
             patch.object(module.importlib.util, "spec_from_file_location", wraps=module.importlib.util.spec_from_file_location) as spec_from_file_location_mock, \
             patch.object(module.importlib, "import_module") as import_module_mock:
            result = module._load_module_without_package_init(module_name)
            self.assertIs(module.sys.modules[module_name], result)

        self.assertEqual(result.LOADED_VALUE, 456)
        self.assertIs(parent_module.childpkg, result)
        self.assertEqual(list(result.__path__), [str(package_root)])
        ensure_package_mock.assert_called_once_with(parent_name)
        spec_from_file_location_mock.assert_called_once_with(
            module_name,
            package_init,
            submodule_search_locations=[str(package_root)],
        )
        import_module_mock.assert_not_called()

    def test_load_module_without_package_init_removes_module_from_sys_modules_when_exec_module_fails(self):
        parent_name = "testpkg"
        module_name = f"{parent_name}.child"
        parent_module = module.types.ModuleType(parent_name)
        module_path = self.root / "site-packages" / parent_name / "child.py"
        module_path.parent.mkdir(parents=True, exist_ok=True)
        module_path.write_text("LOADED_VALUE = 123\n", encoding="utf-8")
        file_spec = type(
            "Spec",
            (),
            {"origin": str(module_path), "submodule_search_locations": None},
        )()
        created_module = module.types.ModuleType(module_name)
        failing_loader = module.types.SimpleNamespace(exec_module=Mock(side_effect=RuntimeError("boom")))
        module_file_spec = module.types.SimpleNamespace(loader=failing_loader)

        with patch.dict(module.sys.modules, {parent_name: parent_module}, clear=False), \
             patch.object(module, "_ensure_virtual_package", return_value=parent_module) as ensure_package_mock, \
             patch.object(module, "_get_module_spec", return_value=file_spec), \
             patch.object(module.importlib.util, "spec_from_file_location", return_value=module_file_spec) as spec_from_file_location_mock, \
             patch.object(module.importlib.util, "module_from_spec", return_value=created_module) as module_from_spec_mock, \
             patch.object(module.importlib, "import_module") as import_module_mock:
            with self.assertRaisesRegex(RuntimeError, "boom"):
                module._load_module_without_package_init(module_name)

            self.assertNotIn(module_name, module.sys.modules)

        ensure_package_mock.assert_called_once_with(parent_name)
        spec_from_file_location_mock.assert_called_once_with(module_name, module_path)
        module_from_spec_mock.assert_called_once_with(module_file_spec)
        failing_loader.exec_module.assert_called_once_with(created_module)
        import_module_mock.assert_not_called()

    def test_load_module_without_package_init_falls_back_to_import_module_when_file_spec_has_no_loader(self):
        parent_name = "testpkg"
        module_name = f"{parent_name}.child"
        parent_module = module.types.ModuleType(parent_name)
        imported_module = object()
        module_path = self.root / "site-packages" / parent_name / "child.py"
        module_path.parent.mkdir(parents=True, exist_ok=True)
        module_path.write_text("LOADED_VALUE = 123\n", encoding="utf-8")
        file_spec = type(
            "Spec",
            (),
            {"origin": str(module_path), "submodule_search_locations": None},
        )()
        module_file_spec = module.types.SimpleNamespace(loader=None)

        with patch.dict(module.sys.modules, {parent_name: parent_module}, clear=False), \
             patch.object(module, "_ensure_virtual_package", return_value=parent_module) as ensure_package_mock, \
             patch.object(module, "_get_module_spec", return_value=file_spec), \
             patch.object(module.importlib.util, "spec_from_file_location", return_value=module_file_spec) as spec_from_file_location_mock, \
             patch.object(module.importlib, "import_module", return_value=imported_module) as import_module_mock:
            result = module._load_module_without_package_init(module_name)

        self.assertIs(result, imported_module)
        self.assertIs(parent_module.child, imported_module)
        ensure_package_mock.assert_called_once_with(parent_name)
        spec_from_file_location_mock.assert_called_once_with(module_name, module_path)
        import_module_mock.assert_called_once_with(module_name)

    def test_ensure_paddlex_ocr_components_shim_returns_early_for_existing_components_shim(self):
        components_package = module.types.ModuleType("paddlex.inference.pipelines.components")
        components_package._chinese_ocr_finder_shim = True

        with patch.object(module, "_ensure_virtual_package", return_value=components_package) as ensure_package_mock, \
             patch.object(module, "_load_module_without_package_init") as load_module_mock:
            module._ensure_paddlex_ocr_components_shim()

        ensure_package_mock.assert_called_once_with("paddlex.inference.pipelines.components")
        load_module_mock.assert_not_called()

    def test_ensure_paddlex_ocr_components_shim_wires_common_exports_to_both_packages(self):
        components_package = module.types.ModuleType("paddlex.inference.pipelines.components")
        common_package = module.types.ModuleType("paddlex.inference.pipelines.components.common")
        crop_by_polys = object()
        sort_poly_boxes = object()
        sort_quad_boxes = object()
        cal_ocr_word_box = object()
        convert_points_to_boxes = object()
        rotate_image = object()
        package_map = {
            "paddlex.inference.pipelines.components": components_package,
            "paddlex.inference.pipelines.components.common": common_package,
        }
        export_modules = {
            "paddlex.inference.pipelines.components.common.crop_image_regions": module.types.SimpleNamespace(CropByPolys=crop_by_polys),
            "paddlex.inference.pipelines.components.common.sort_boxes": module.types.SimpleNamespace(
                SortPolyBoxes=sort_poly_boxes,
                SortQuadBoxes=sort_quad_boxes,
            ),
            "paddlex.inference.pipelines.components.common.cal_ocr_word_box": module.types.SimpleNamespace(
                cal_ocr_word_box=cal_ocr_word_box,
            ),
            "paddlex.inference.pipelines.components.common.convert_points_and_boxes": module.types.SimpleNamespace(
                convert_points_to_boxes=convert_points_to_boxes,
            ),
            "paddlex.inference.pipelines.components.common.warp_image": module.types.SimpleNamespace(
                rotate_image=rotate_image,
            ),
        }

        with patch.object(module, "_ensure_virtual_package", side_effect=lambda package_name: package_map[package_name]), \
             patch.object(module, "_load_module_without_package_init", side_effect=lambda module_name: export_modules[module_name]) as load_module_mock:
            module._ensure_paddlex_ocr_components_shim()

        expected_exports = {
            "CropByPolys": crop_by_polys,
            "SortPolyBoxes": sort_poly_boxes,
            "SortQuadBoxes": sort_quad_boxes,
            "cal_ocr_word_box": cal_ocr_word_box,
            "convert_points_to_boxes": convert_points_to_boxes,
            "rotate_image": rotate_image,
        }
        for export_name, export_value in expected_exports.items():
            self.assertIs(getattr(common_package, export_name), export_value)
            self.assertIs(getattr(components_package, export_name), export_value)

        self.assertTrue(common_package._chinese_ocr_finder_shim)
        self.assertTrue(components_package._chinese_ocr_finder_shim)
        self.assertEqual(
            [call.args[0] for call in load_module_mock.call_args_list],
            [
                "paddlex.inference.pipelines.components.common.crop_image_regions",
                "paddlex.inference.pipelines.components.common.sort_boxes",
                "paddlex.inference.pipelines.components.common.sort_boxes",
                "paddlex.inference.pipelines.components.common.cal_ocr_word_box",
                "paddlex.inference.pipelines.components.common.convert_points_and_boxes",
                "paddlex.inference.pipelines.components.common.warp_image",
            ],
        )

    def test_load_paddleocr_class_installs_shim_and_returns_module_attribute(self):
        fake_paddleocr_class = type("FakePaddleOCR", (), {})
        fake_module = type("PaddleOCRModule", (), {"PaddleOCR": fake_paddleocr_class})()

        with patch.object(module, "_ensure_paddlex_shim") as ensure_shim_mock, \
             patch.object(module, "_load_module_without_package_init", return_value=fake_module) as load_module_mock:
            self.assertIs(module._load_paddleocr_class(), fake_paddleocr_class)

        ensure_shim_mock.assert_called_once_with()
        load_module_mock.assert_called_once_with("paddleocr._pipelines.ocr")

    def test_load_paddleocr_class_loads_runtime_module_from_file_spec(self):
        package_root = self.root / "site-packages" / "paddleocr"
        pipelines_root = package_root / "_pipelines"
        module_path = pipelines_root / "ocr.py"
        (package_root / "__init__.py").parent.mkdir(parents=True, exist_ok=True)
        (package_root / "__init__.py").write_text("", encoding="utf-8")
        (pipelines_root / "__init__.py").parent.mkdir(parents=True, exist_ok=True)
        (pipelines_root / "__init__.py").write_text("", encoding="utf-8")
        module_path.write_text("class PaddleOCR:\n    pass\n", encoding="utf-8")
        spec_map = {
            "paddleocr": type(
                "Spec",
                (),
                {"origin": str(package_root / "__init__.py"), "submodule_search_locations": [str(package_root)]},
            )(),
            "paddleocr._pipelines": type(
                "Spec",
                (),
                {"origin": str(pipelines_root / "__init__.py"), "submodule_search_locations": [str(pipelines_root)]},
            )(),
            "paddleocr._pipelines.ocr": type(
                "Spec",
                (),
                {"origin": str(module_path), "submodule_search_locations": None},
            )(),
        }

        with patch.dict(
            module.sys.modules,
            {name: None for name in spec_map},
            clear=False,
        ), \
             patch.object(module, "_ensure_paddlex_shim") as ensure_shim_mock, \
             patch.object(module, "_get_module_spec", side_effect=lambda name: spec_map[name]), \
             patch.object(module.importlib, "import_module") as import_module_mock:
            paddleocr_class = module._load_paddleocr_class()
            loaded_module = module.sys.modules["paddleocr._pipelines.ocr"]
            parent_module = module.sys.modules["paddleocr._pipelines"]

            self.assertIs(paddleocr_class, loaded_module.PaddleOCR)
            self.assertEqual(paddleocr_class.__module__, "paddleocr._pipelines.ocr")
            self.assertIs(parent_module.ocr, loaded_module)

        ensure_shim_mock.assert_called_once_with()
        import_module_mock.assert_not_called()

    def test_load_paddleocr_class_uses_packaged_runtime_fallback_for_frozen_spec(self):
        runtime_root = self.root / "release"
        fake_paddleocr_class = type("FakePaddleOCR", (), {})
        packaged_spec = type(
            "Spec",
            (),
            {"origin": "frozen", "submodule_search_locations": None},
        )()
        spec_map = {
            "paddleocr": packaged_spec,
            "paddleocr._pipelines": packaged_spec,
            "paddleocr._pipelines.ocr": packaged_spec,
        }

        def fake_import_module(module_name: str):
            imported_module = module.types.ModuleType(module_name)
            imported_module.PaddleOCR = fake_paddleocr_class
            module.sys.modules[module_name] = imported_module
            return imported_module

        with patch.dict(
            module.sys.modules,
            {name: None for name in spec_map},
            clear=False,
        ), \
             patch.object(module, "_ensure_paddlex_shim") as ensure_shim_mock, \
             patch.object(module, "_get_module_spec", side_effect=lambda name: spec_map[name]), \
             patch.object(module, "is_packaged_runtime", return_value=True), \
             patch.object(module, "get_runtime_resource_dir", return_value=runtime_root), \
             patch.object(module.importlib, "import_module", side_effect=fake_import_module) as import_module_mock:
            paddleocr_class = module._load_paddleocr_class()
            package_module = module.sys.modules["paddleocr"]
            parent_module = module.sys.modules["paddleocr._pipelines"]
            loaded_module = module.sys.modules["paddleocr._pipelines.ocr"]

        self.assertIs(paddleocr_class, fake_paddleocr_class)
        self.assertIs(package_module._pipelines, parent_module)
        self.assertEqual(package_module.__path__, [str(runtime_root / "paddleocr")])
        self.assertEqual(parent_module.__path__, [str(runtime_root / "paddleocr" / "_pipelines")])
        self.assertIs(parent_module.ocr, loaded_module)
        self.assertIs(loaded_module.PaddleOCR, fake_paddleocr_class)
        ensure_shim_mock.assert_called_once_with()
        import_module_mock.assert_called_once_with("paddleocr._pipelines.ocr")

    def test_load_pdf_reader_class_installs_shim_and_returns_module_attribute(self):
        fake_pdf_reader_class = type("FakePDFReader", (), {})
        fake_module = type("PDFReaderModule", (), {"PDFReader": fake_pdf_reader_class})()

        with patch.object(module, "_ensure_paddlex_shim") as ensure_shim_mock, \
             patch.object(module, "_load_module_without_package_init", return_value=fake_module) as load_module_mock:
            self.assertIs(module._load_pdf_reader_class(), fake_pdf_reader_class)

        ensure_shim_mock.assert_called_once_with()
        load_module_mock.assert_called_once_with("paddlex.inference.utils.io.readers")

    def test_load_pdf_reader_class_uses_packaged_runtime_fallback_for_frozen_spec(self):
        runtime_root = self.root / "release"
        fake_pdf_reader_class = type("FakePDFReader", (), {})
        packaged_spec = type(
            "Spec",
            (),
            {"origin": "frozen", "submodule_search_locations": None},
        )()
        spec_map = {
            "paddlex": packaged_spec,
            "paddlex.inference": packaged_spec,
            "paddlex.inference.utils": packaged_spec,
            "paddlex.inference.utils.io": packaged_spec,
            "paddlex.inference.utils.io.readers": packaged_spec,
        }

        def fake_import_module(module_name: str):
            imported_module = module.types.ModuleType(module_name)
            imported_module.PDFReader = fake_pdf_reader_class
            module.sys.modules[module_name] = imported_module
            return imported_module

        with patch.dict(
            module.sys.modules,
            {name: None for name in spec_map},
            clear=False,
        ), \
             patch.object(module, "_ensure_paddlex_shim") as ensure_shim_mock, \
             patch.object(module, "_get_module_spec", side_effect=lambda name: spec_map[name]), \
             patch.object(module, "is_packaged_runtime", return_value=True), \
             patch.object(module, "get_runtime_resource_dir", return_value=runtime_root), \
             patch.object(module.importlib, "import_module", side_effect=fake_import_module) as import_module_mock:
            pdf_reader_class = module._load_pdf_reader_class()
            package_module = module.sys.modules["paddlex"]
            inference_module = module.sys.modules["paddlex.inference"]
            utils_module = module.sys.modules["paddlex.inference.utils"]
            io_module = module.sys.modules["paddlex.inference.utils.io"]
            loaded_module = module.sys.modules["paddlex.inference.utils.io.readers"]

        self.assertIs(pdf_reader_class, fake_pdf_reader_class)
        self.assertIs(package_module.inference, inference_module)
        self.assertIs(inference_module.utils, utils_module)
        self.assertIs(utils_module.io, io_module)
        self.assertEqual(package_module.__path__, [str(runtime_root / "paddlex")])
        self.assertEqual(inference_module.__path__, [str(runtime_root / "paddlex" / "inference")])
        self.assertEqual(utils_module.__path__, [str(runtime_root / "paddlex" / "inference" / "utils")])
        self.assertEqual(io_module.__path__, [str(runtime_root / "paddlex" / "inference" / "utils" / "io")])
        self.assertIs(io_module.readers, loaded_module)
        self.assertIs(loaded_module.PDFReader, fake_pdf_reader_class)
        ensure_shim_mock.assert_called_once_with()
        import_module_mock.assert_called_once_with("paddlex.inference.utils.io.readers")

    def test_load_pdf_reader_class_loads_runtime_module_from_file_spec(self):
        package_root = self.root / "site-packages" / "paddlex"
        inference_root = package_root / "inference"
        utils_root = inference_root / "utils"
        io_root = utils_root / "io"
        module_path = io_root / "readers.py"
        for init_path in (
            package_root / "__init__.py",
            inference_root / "__init__.py",
            utils_root / "__init__.py",
            io_root / "__init__.py",
        ):
            init_path.parent.mkdir(parents=True, exist_ok=True)
            init_path.write_text("", encoding="utf-8")
        module_path.write_text("class PDFReader:\n    pass\n", encoding="utf-8")
        spec_map = {
            "paddlex": type(
                "Spec",
                (),
                {"origin": str(package_root / "__init__.py"), "submodule_search_locations": [str(package_root)]},
            )(),
            "paddlex.inference": type(
                "Spec",
                (),
                {"origin": str(inference_root / "__init__.py"), "submodule_search_locations": [str(inference_root)]},
            )(),
            "paddlex.inference.utils": type(
                "Spec",
                (),
                {"origin": str(utils_root / "__init__.py"), "submodule_search_locations": [str(utils_root)]},
            )(),
            "paddlex.inference.utils.io": type(
                "Spec",
                (),
                {"origin": str(io_root / "__init__.py"), "submodule_search_locations": [str(io_root)]},
            )(),
            "paddlex.inference.utils.io.readers": type(
                "Spec",
                (),
                {"origin": str(module_path), "submodule_search_locations": None},
            )(),
        }

        with patch.dict(
            module.sys.modules,
            {name: None for name in spec_map},
            clear=False,
        ), \
             patch.object(module, "_ensure_paddlex_shim") as ensure_shim_mock, \
             patch.object(module, "_get_module_spec", side_effect=lambda name: spec_map[name]), \
             patch.object(module.importlib, "import_module") as import_module_mock:
            pdf_reader_class = module._load_pdf_reader_class()
            loaded_module = module.sys.modules["paddlex.inference.utils.io.readers"]
            parent_module = module.sys.modules["paddlex.inference.utils.io"]

            self.assertIs(pdf_reader_class, loaded_module.PDFReader)
            self.assertEqual(pdf_reader_class.__module__, "paddlex.inference.utils.io.readers")
            self.assertIs(parent_module.readers, loaded_module)

        ensure_shim_mock.assert_called_once_with()
        import_module_mock.assert_not_called()

    def test_install_local_paddlex_pp_option_shim_creates_and_registers_shim_module(self):
        utils_package = module.types.ModuleType("paddlex.inference.utils")
        module_name = "paddlex.inference.utils.pp_option"
        extra_stubs = {
            "paddlex": module.types.ModuleType("paddlex"),
            "paddlex.inference": module.types.ModuleType("paddlex.inference"),
            "paddlex.inference.utils": utils_package,
        }

        with patch.dict(module.sys.modules, {**extra_stubs, module_name: None}, clear=False):
            result = module._install_local_paddlex_pp_option_shim()
            installed_module = module.sys.modules[module_name]

        self.assertIs(result, module._LocalPaddlePredictorOption)
        self.assertIs(installed_module.PaddlePredictorOption, module._LocalPaddlePredictorOption)
        self.assertIs(installed_module.get_default_device, module.get_default_device)
        self.assertIs(installed_module.parse_device, module.parse_device)
        self.assertIs(installed_module.set_env_for_device_type, module.set_env_for_device_type)
        self.assertIs(installed_module.get_default_run_mode, module.get_default_run_mode)
        self.assertTrue(installed_module._chinese_ocr_finder_shim)
        self.assertEqual(installed_module.__package__, "paddlex.inference.utils")
        self.assertEqual(installed_module.__file__, str(Path(module.__file__).resolve()))
        self.assertEqual(installed_module.__spec__.name, module_name)
        self.assertIsNone(installed_module.__spec__.loader)
        self.assertIs(utils_package.pp_option, installed_module)

    def test_install_local_paddlex_pp_option_shim_reuses_existing_shim_module(self):
        utils_package = module.types.ModuleType("paddlex.inference.utils")
        module_name = "paddlex.inference.utils.pp_option"
        existing_option_class = type("ExistingPaddlePredictorOption", (), {})
        existing_module = module.types.ModuleType(module_name)
        existing_module.PaddlePredictorOption = existing_option_class
        existing_module._chinese_ocr_finder_shim = True
        extra_stubs = {
            "paddlex": module.types.ModuleType("paddlex"),
            "paddlex.inference": module.types.ModuleType("paddlex.inference"),
            "paddlex.inference.utils": utils_package,
        }

        with patch.dict(module.sys.modules, {**extra_stubs, module_name: existing_module}, clear=False):
            result = module._install_local_paddlex_pp_option_shim()
            installed_module = module.sys.modules[module_name]

        self.assertIs(result, existing_option_class)
        self.assertIs(installed_module, existing_module)
        self.assertIs(installed_module.get_default_device, module.get_default_device)
        self.assertIs(installed_module.parse_device, module.parse_device)
        self.assertIs(installed_module.set_env_for_device_type, module.set_env_for_device_type)
        self.assertIs(installed_module.get_default_run_mode, module.get_default_run_mode)
        self.assertIs(utils_package.pp_option, existing_module)

    def test_install_local_paddlex_pp_option_shim_reuses_existing_shim_module_with_mismatched_spec(self):
        utils_package = module.types.ModuleType("paddlex.inference.utils")
        module_name = "paddlex.inference.utils.pp_option"
        existing_option_class = type("ExistingPaddlePredictorOption", (), {})
        existing_module = module.types.ModuleType(module_name)
        existing_module.PaddlePredictorOption = existing_option_class
        existing_module._chinese_ocr_finder_shim = True
        existing_module.__spec__ = module.importlib.machinery.ModuleSpec("other.module", loader=object())
        extra_stubs = {
            "paddlex": module.types.ModuleType("paddlex"),
            "paddlex.inference": module.types.ModuleType("paddlex.inference"),
            "paddlex.inference.utils": utils_package,
        }

        with patch.dict(module.sys.modules, {**extra_stubs, module_name: existing_module}, clear=False):
            result = module._install_local_paddlex_pp_option_shim()
            installed_module = module.sys.modules[module_name]

        self.assertIs(result, existing_option_class)
        self.assertIs(installed_module, existing_module)
        self.assertEqual(installed_module.__spec__.name, module_name)
        self.assertIsNone(installed_module.__spec__.loader)
        self.assertIs(utils_package.pp_option, existing_module)

    def test_install_local_paddlex_pp_option_shim_replaces_existing_non_shim_module(self):
        utils_package = module.types.ModuleType("paddlex.inference.utils")
        module_name = "paddlex.inference.utils.pp_option"
        existing_module = module.types.ModuleType(module_name)
        existing_module.PaddlePredictorOption = type("ForeignPaddlePredictorOption", (), {})
        extra_stubs = {
            "paddlex": module.types.ModuleType("paddlex"),
            "paddlex.inference": module.types.ModuleType("paddlex.inference"),
            "paddlex.inference.utils": utils_package,
        }

        with patch.dict(module.sys.modules, {**extra_stubs, module_name: existing_module}, clear=False):
            result = module._install_local_paddlex_pp_option_shim()
            installed_module = module.sys.modules[module_name]

        self.assertIs(result, module._LocalPaddlePredictorOption)
        self.assertIsNot(installed_module, existing_module)
        self.assertIs(installed_module.PaddlePredictorOption, module._LocalPaddlePredictorOption)
        self.assertTrue(installed_module._chinese_ocr_finder_shim)
        self.assertEqual(installed_module.__package__, "paddlex.inference.utils")
        self.assertEqual(installed_module.__file__, str(Path(module.__file__).resolve()))
        self.assertEqual(installed_module.__spec__.name, module_name)
        self.assertIsNone(installed_module.__spec__.loader)
        self.assertIs(utils_package.pp_option, installed_module)

    def test_bootstrap_local_paddlex_pp_option_shim_installs_shim_module(self):
        with patch.object(module, "_install_local_paddlex_pp_option_shim") as install_pp_option_mock:
            module._bootstrap_local_paddlex_pp_option_shim()

        install_pp_option_mock.assert_called_once_with()

    def test_bootstrap_local_paddlex_pp_option_shim_ignores_import_error(self):
        with patch.object(module, "_install_local_paddlex_pp_option_shim", side_effect=ImportError("missing paddlex")) as install_pp_option_mock:
            module._bootstrap_local_paddlex_pp_option_shim()

        install_pp_option_mock.assert_called_once_with()

    def test_ensure_paddlex_ocr_components_shim_installs_common_exports_on_both_packages(self):
        components_package = module.types.ModuleType("paddlex.inference.pipelines.components")
        common_package = module.types.ModuleType("paddlex.inference.pipelines.components.common")
        crop_by_polys = object()
        sort_poly_boxes = object()
        sort_quad_boxes = object()
        cal_ocr_word_box = object()
        convert_points_to_boxes = object()
        rotate_image = object()
        package_map = {
            "paddlex.inference.pipelines.components": components_package,
            "paddlex.inference.pipelines.components.common": common_package,
        }
        module_map = {
            "paddlex.inference.pipelines.components.common.crop_image_regions": type(
                "CropImageRegionsModule",
                (),
                {"CropByPolys": crop_by_polys},
            )(),
            "paddlex.inference.pipelines.components.common.sort_boxes": type(
                "SortBoxesModule",
                (),
                {"SortPolyBoxes": sort_poly_boxes, "SortQuadBoxes": sort_quad_boxes},
            )(),
            "paddlex.inference.pipelines.components.common.cal_ocr_word_box": type(
                "CalOcrWordBoxModule",
                (),
                {"cal_ocr_word_box": cal_ocr_word_box},
            )(),
            "paddlex.inference.pipelines.components.common.convert_points_and_boxes": type(
                "ConvertPointsAndBoxesModule",
                (),
                {"convert_points_to_boxes": convert_points_to_boxes},
            )(),
            "paddlex.inference.pipelines.components.common.warp_image": type(
                "WarpImageModule",
                (),
                {"rotate_image": rotate_image},
            )(),
        }

        def fake_ensure_virtual_package(package_name: str):
            return package_map[package_name]

        load_module_mock = Mock(side_effect=lambda module_name: module_map[module_name])

        with patch.object(module, "_ensure_virtual_package", side_effect=fake_ensure_virtual_package) as ensure_package_mock, \
             patch.object(module, "_load_module_without_package_init", load_module_mock):
            module._ensure_paddlex_ocr_components_shim()

        expected_exports = {
            "CropByPolys": crop_by_polys,
            "SortPolyBoxes": sort_poly_boxes,
            "SortQuadBoxes": sort_quad_boxes,
            "cal_ocr_word_box": cal_ocr_word_box,
            "convert_points_to_boxes": convert_points_to_boxes,
            "rotate_image": rotate_image,
        }
        for export_name, export_value in expected_exports.items():
            self.assertIs(getattr(common_package, export_name), export_value)
            self.assertIs(getattr(components_package, export_name), export_value)

        self.assertTrue(common_package._chinese_ocr_finder_shim)
        self.assertTrue(components_package._chinese_ocr_finder_shim)
        self.assertEqual(
            [call.args[0] for call in ensure_package_mock.call_args_list],
            [
                "paddlex.inference.pipelines.components",
                "paddlex.inference.pipelines.components.common",
            ],
        )
        self.assertEqual(
            [call.args[0] for call in load_module_mock.call_args_list],
            [
                "paddlex.inference.pipelines.components.common.crop_image_regions",
                "paddlex.inference.pipelines.components.common.sort_boxes",
                "paddlex.inference.pipelines.components.common.sort_boxes",
                "paddlex.inference.pipelines.components.common.cal_ocr_word_box",
                "paddlex.inference.pipelines.components.common.convert_points_and_boxes",
                "paddlex.inference.pipelines.components.common.warp_image",
            ],
        )

    def test_ensure_paddlex_shim_returns_early_for_existing_inference_shim(self):
        fake_paddlex = module.types.ModuleType("paddlex")
        fake_inference = module.types.ModuleType("paddlex.inference")
        fake_inference._chinese_ocr_finder_shim = True

        with patch.object(module, "_ensure_virtual_package", side_effect=[fake_paddlex, fake_inference]) as ensure_package_mock, \
             patch.object(module, "_install_local_paddlex_pp_option_shim") as install_pp_option_mock, \
             patch.object(module, "_load_module_without_package_init") as load_module_mock:
            module._ensure_paddlex_shim()

        self.assertEqual(
            [call.args[0] for call in ensure_package_mock.call_args_list],
            ["paddlex", "paddlex.inference"],
        )
        install_pp_option_mock.assert_not_called()
        load_module_mock.assert_not_called()

    def test_ensure_paddlex_shim_uses_minimal_ocr_config_without_loading_yaml(self):
        fake_paddlex = module.types.ModuleType("paddlex")
        fake_inference = module.types.ModuleType("paddlex.inference")
        fake_models = module.types.ModuleType("paddlex.inference.models")
        fake_models_base = module.types.ModuleType("paddlex.inference.models.base")
        fake_pipelines = module.types.ModuleType("paddlex.inference.pipelines")
        fake_utils = module.types.ModuleType("paddlex.inference.utils")
        fake_base_predictor_class = type("FakeBasePredictor", (), {})
        fake_base_predictor_module = type(
            "BasePredictorModule",
            (),
            {"BasePredictor": fake_base_predictor_class},
        )()
        fake_base_pipeline_class = type("FakeBasePipeline", (), {})
        fake_base_pipeline_module = type(
            "BasePipelineModule",
            (),
            {"BasePipeline": fake_base_pipeline_class},
        )()

        package_map = {
            "paddlex": fake_paddlex,
            "paddlex.inference": fake_inference,
            "paddlex.inference.models": fake_models,
            "paddlex.inference.models.base": fake_models_base,
            "paddlex.inference.pipelines": fake_pipelines,
            "paddlex.inference.utils": fake_utils,
        }
        sys_modules_map = {**package_map, "paddlex.inference.utils.pp_option": None}
        for package in package_map.values():
            package.__path__ = []
        fake_paddlex.inference = fake_inference
        fake_inference.models = fake_models
        fake_inference.pipelines = fake_pipelines
        fake_inference.utils = fake_utils
        fake_models.base = fake_models_base

        def fake_ensure_virtual_package(package_name: str):
            return package_map[package_name]

        def fake_load_module_without_package_init(module_name: str):
            if module_name == "paddlex.inference.models.base.predictor.base_predictor":
                return fake_base_predictor_module
            if module_name == "paddlex.inference.pipelines.base":
                return fake_base_pipeline_module
            raise AssertionError(f"Unexpected module load: {module_name}")

        load_module_mock = Mock(side_effect=fake_load_module_without_package_init)

        with patch.dict(module.sys.modules, sys_modules_map, clear=False), \
             patch.object(module, "_ensure_virtual_package", side_effect=fake_ensure_virtual_package), \
             patch.object(module, "_load_module_without_package_init", load_module_mock):
            module._ensure_paddlex_shim()
            config = fake_inference.load_pipeline_config("OCR")
            installed_pp_option = module.sys.modules["paddlex.inference.utils.pp_option"]

        self.assertEqual(config, module._get_minimal_ocr_pipeline_config())
        self.assertIs(fake_inference.PaddlePredictorOption, module._LocalPaddlePredictorOption)
        self.assertIs(fake_utils.pp_option, installed_pp_option)
        self.assertIs(fake_utils.pp_option.PaddlePredictorOption, module._LocalPaddlePredictorOption)
        self.assertIs(fake_utils.pp_option.get_default_device, module.get_default_device)
        self.assertIs(fake_utils.pp_option.parse_device, module.parse_device)
        self.assertIs(fake_utils.pp_option.set_env_for_device_type, module.set_env_for_device_type)
        self.assertIs(fake_utils.pp_option.get_default_run_mode, module.get_default_run_mode)
        self.assertIs(installed_pp_option.get_default_run_mode, module.get_default_run_mode)
        self.assertTrue(fake_utils.pp_option._chinese_ocr_finder_shim)
        self.assertNotIn(
            "paddlex.utils.config",
            [call.args[0] for call in load_module_mock.call_args_list],
        )
        self.assertNotIn(
            "paddlex.inference.utils.pp_option",
            [call.args[0] for call in load_module_mock.call_args_list],
        )

    def test_ensure_paddlex_shim_supports_packaged_style_downstream_pp_option_imports(self):
        fake_paddlex = module.types.ModuleType("paddlex")
        fake_inference = module.types.ModuleType("paddlex.inference")
        fake_models = module.types.ModuleType("paddlex.inference.models")
        fake_models_base = module.types.ModuleType("paddlex.inference.models.base")
        fake_models_common = module.types.ModuleType("paddlex.inference.models.common")
        fake_pipelines = module.types.ModuleType("paddlex.inference.pipelines")
        fake_utils = module.types.ModuleType("paddlex.inference.utils")
        fake_base_predictor_class = type("FakeBasePredictor", (), {})
        fake_base_predictor_module = type(
            "BasePredictorModule",
            (),
            {"BasePredictor": fake_base_predictor_class},
        )()
        fake_base_pipeline_class = type("FakeBasePipeline", (), {})
        fake_base_pipeline_module = type(
            "BasePipelineModule",
            (),
            {"BasePipeline": fake_base_pipeline_class},
        )()
        package_map = {
            "paddlex": fake_paddlex,
            "paddlex.inference": fake_inference,
            "paddlex.inference.models": fake_models,
            "paddlex.inference.models.base": fake_models_base,
            "paddlex.inference.models.common": fake_models_common,
            "paddlex.inference.pipelines": fake_pipelines,
            "paddlex.inference.utils": fake_utils,
        }
        sys_modules_map = {
            **package_map,
            "paddlex.inference.utils.pp_option": None,
            "paddlex.inference.models.common.static_infer": None,
        }
        for package in package_map.values():
            package.__path__ = []
        fake_paddlex.inference = fake_inference
        fake_inference.models = fake_models
        fake_inference.pipelines = fake_pipelines
        fake_inference.utils = fake_utils
        fake_models.base = fake_models_base
        fake_models.common = fake_models_common

        static_infer_path = self.root / "site-packages" / "paddlex" / "inference" / "models" / "common" / "static_infer.py"
        static_infer_path.parent.mkdir(parents=True, exist_ok=True)
        static_infer_path.write_text(
            "from paddlex.inference.utils.pp_option import get_default_run_mode\n"
            "IMPORTED_RUN_MODE = get_default_run_mode('ocr-model', 'cpu')\n"
            "IMPORTED_FUNCTION = get_default_run_mode\n",
            encoding="utf-8",
        )
        static_infer_spec = type(
            "Spec",
            (),
            {"origin": str(static_infer_path), "submodule_search_locations": None},
        )()
        original_load_module = module._load_module_without_package_init

        def fake_ensure_virtual_package(package_name: str):
            return package_map[package_name]

        def fake_load_module_without_package_init(module_name: str):
            if module_name == "paddlex.inference.models.base.predictor.base_predictor":
                return fake_base_predictor_module
            if module_name == "paddlex.inference.pipelines.base":
                return fake_base_pipeline_module
            return original_load_module(module_name)

        with patch.dict(module.sys.modules, sys_modules_map, clear=False), \
             patch.object(module, "_ensure_virtual_package", side_effect=fake_ensure_virtual_package), \
             patch.object(module, "_get_module_spec", side_effect=lambda module_name: static_infer_spec), \
             patch.object(module, "_load_module_without_package_init", side_effect=fake_load_module_without_package_init):
            module._ensure_paddlex_shim()
            loaded_module = original_load_module("paddlex.inference.models.common.static_infer")
            installed_pp_option = module.sys.modules["paddlex.inference.utils.pp_option"]

        self.assertEqual(loaded_module.IMPORTED_RUN_MODE, "paddle")
        self.assertIs(loaded_module.IMPORTED_FUNCTION, module.get_default_run_mode)
        self.assertIs(fake_models_common.static_infer, loaded_module)
        self.assertIs(installed_pp_option.get_default_run_mode, module.get_default_run_mode)

    def test_get_minimal_ocr_pipeline_config_disables_extra_processing_and_uses_local_models(self):
        config = module._get_minimal_ocr_pipeline_config()

        self.assertEqual(config["pipeline_name"], "OCR")
        self.assertEqual(config["text_type"], "general")
        self.assertFalse(config["use_doc_preprocessor"])
        self.assertFalse(config["use_textline_orientation"])
        self.assertEqual(
            config["SubModules"]["TextDetection"]["model_name"],
            module.TEXT_DETECTION_MODEL_NAME,
        )
        self.assertIsNone(config["SubModules"]["TextDetection"]["model_dir"])
        self.assertEqual(
            config["SubModules"]["TextRecognition"]["model_name"],
            module.TEXT_RECOGNITION_MODEL_NAME,
        )
        self.assertIsNone(config["SubModules"]["TextRecognition"]["model_dir"])

    def test_resolve_pipeline_config_path_prefers_bundled_runtime_named_pipeline(self):
        resource_root = self.root / "resources"
        bundled_path = resource_root / "paddlex" / "configs" / "pipelines" / "Custom.yaml"
        bundled_path.parent.mkdir(parents=True, exist_ok=True)
        bundled_path.write_text("pipeline_name: OCR\n", encoding="utf-8")
        package_root = self.root / "site-packages" / "paddlex"
        package_path = package_root / "configs" / "pipelines" / "Custom.yaml"
        package_path.parent.mkdir(parents=True, exist_ok=True)
        package_path.write_text("pipeline_name: OCR\n", encoding="utf-8")

        with patch.object(module, "get_runtime_resource_dir", return_value=resource_root), \
             patch.object(module, "_get_package_root", return_value=package_root):
            self.assertEqual(module._resolve_pipeline_config_path("Custom"), bundled_path.resolve())

    def test_resolve_pipeline_config_path_falls_back_to_package_root_named_pipeline(self):
        resource_root = self.root / "resources"
        package_root = self.root / "site-packages" / "paddlex"
        package_path = package_root / "configs" / "pipelines" / "Custom.yaml"
        package_path.parent.mkdir(parents=True, exist_ok=True)
        package_path.write_text("pipeline_name: OCR\n", encoding="utf-8")

        with patch.object(module, "get_runtime_resource_dir", return_value=resource_root), \
             patch.object(module, "_get_package_root", return_value=package_root):
            self.assertEqual(module._resolve_pipeline_config_path("Custom"), package_path.resolve())

    def test_resolve_pipeline_config_path_raises_for_missing_named_pipeline(self):
        with patch.object(module, "_get_bundled_pipeline_config_path", return_value=None) as bundled_path_mock, \
             patch.object(module, "_get_package_root", return_value=None) as package_root_mock:
            with self.assertRaisesRegex(Exception, "Please use a pipeline name or a config file path!"):
                module._resolve_pipeline_config_path("MissingPipeline")

        bundled_path_mock.assert_called_once_with("MissingPipeline")
        package_root_mock.assert_called_once_with("paddlex")

    def test_resolve_pipeline_config_path_checks_runtime_resources_for_relative_yaml(self):
        resource_root = self.root / "resources"
        bundled_path = resource_root / "paddlex" / "configs" / "pipelines" / "Custom.yaml"
        bundled_path.parent.mkdir(parents=True, exist_ok=True)
        bundled_path.write_text("pipeline_name: OCR\n", encoding="utf-8")

        with patch.object(module, "get_runtime_resource_dir", return_value=resource_root), \
             patch.object(module, "_get_package_root", return_value=None):
            self.assertEqual(module._resolve_pipeline_config_path("Custom.yaml"), bundled_path.resolve())

    def test_resolve_pipeline_config_path_checks_runtime_root_for_relative_yaml(self):
        resource_root = self.root / "resources"
        bundled_path = resource_root / "nested" / "Custom.yaml"
        bundled_path.parent.mkdir(parents=True, exist_ok=True)
        bundled_path.write_text("pipeline_name: OCR\n", encoding="utf-8")

        with patch.object(module, "get_runtime_resource_dir", return_value=resource_root), \
             patch.object(module, "_get_package_root", return_value=None):
            self.assertEqual(module._resolve_pipeline_config_path("nested/Custom.yaml"), bundled_path.resolve())

    def test_resolve_pipeline_config_path_falls_back_to_package_root_pipeline_dir_for_relative_yaml(self):
        resource_root = self.root / "resources"
        package_root = self.root / "site-packages" / "paddlex"
        package_path = package_root / "configs" / "pipelines" / "Custom.yaml"
        package_path.parent.mkdir(parents=True, exist_ok=True)
        package_path.write_text("pipeline_name: OCR\n", encoding="utf-8")

        with patch.object(module, "get_runtime_resource_dir", return_value=resource_root), \
             patch.object(module, "_get_package_root", return_value=package_root):
            self.assertEqual(module._resolve_pipeline_config_path("Custom.yaml"), package_path.resolve())

    def test_resolve_pipeline_config_path_falls_back_to_package_root_relative_yaml(self):
        resource_root = self.root / "resources"
        package_root = self.root / "site-packages" / "paddlex"
        package_path = package_root / "nested" / "Custom.yaml"
        package_path.parent.mkdir(parents=True, exist_ok=True)
        package_path.write_text("pipeline_name: OCR\n", encoding="utf-8")

        with patch.object(module, "get_runtime_resource_dir", return_value=resource_root), \
             patch.object(module, "_get_package_root", return_value=package_root):
            self.assertEqual(module._resolve_pipeline_config_path("nested/Custom.yaml"), package_path.resolve())

    def test_resolve_pipeline_config_path_raises_for_missing_relative_yaml_path(self):
        resource_root = self.root / "resources"
        package_root = self.root / "site-packages" / "paddlex"
        package_root.mkdir(parents=True, exist_ok=True)

        with patch.object(module, "get_runtime_resource_dir", return_value=resource_root) as runtime_dir_mock, \
             patch.object(module, "_get_package_root", return_value=package_root) as package_root_mock:
            with self.assertRaisesRegex(Exception, "Please use a pipeline name or a config file path!"):
                module._resolve_pipeline_config_path("nested/Missing.yaml")

        runtime_dir_mock.assert_called_once_with()
        package_root_mock.assert_called_once_with("paddlex")

    def test_get_bundled_pipeline_config_path_returns_none_when_runtime_file_is_missing(self):
        resource_root = self.root / "resources"

        with patch.object(module, "get_runtime_resource_dir", return_value=resource_root):
            self.assertIsNone(module._get_bundled_pipeline_config_path("OCR"))

    def test_get_bundled_pipeline_config_path_returns_resolved_runtime_file(self):
        resource_root = self.root / "resources"
        bundled_path = resource_root / "paddlex" / "configs" / "pipelines" / "OCR.yaml"
        bundled_path.parent.mkdir(parents=True, exist_ok=True)
        bundled_path.write_text("pipeline_name: OCR\n", encoding="utf-8")

        with patch.object(module, "get_runtime_resource_dir", return_value=resource_root):
            self.assertEqual(module._get_bundled_pipeline_config_path("OCR"), bundled_path.resolve())

    def test_resolve_pipeline_config_path_returns_resolved_absolute_yaml_path(self):
        absolute_path = self.root / "configs" / "Custom.yaml"
        absolute_path.parent.mkdir(parents=True, exist_ok=True)
        absolute_path.write_text("pipeline_name: OCR\n", encoding="utf-8")

        with patch.object(module, "get_runtime_resource_dir") as runtime_dir_mock, \
             patch.object(module, "_get_package_root") as package_root_mock:
            self.assertEqual(module._resolve_pipeline_config_path(str(absolute_path.resolve())), absolute_path.resolve())

        runtime_dir_mock.assert_not_called()
        package_root_mock.assert_not_called()

    def test_resolve_pipeline_config_path_raises_for_missing_absolute_yaml_path(self):
        absolute_path = (self.root / "configs" / "Missing.yaml").resolve()

        with patch.object(module, "get_runtime_resource_dir") as runtime_dir_mock, \
             patch.object(module, "_get_package_root") as package_root_mock:
            with self.assertRaisesRegex(Exception, "Please use a pipeline name or a config file path!"):
                module._resolve_pipeline_config_path(str(absolute_path))

        runtime_dir_mock.assert_not_called()
        package_root_mock.assert_not_called()

    def test_ensure_paddlex_shim_uses_resolved_pipeline_path_for_yaml_loading(self):
        fake_paddlex = module.types.ModuleType("paddlex")
        fake_inference = module.types.ModuleType("paddlex.inference")
        fake_models = module.types.ModuleType("paddlex.inference.models")
        fake_models_base = module.types.ModuleType("paddlex.inference.models.base")
        fake_pipelines = module.types.ModuleType("paddlex.inference.pipelines")
        fake_utils = module.types.ModuleType("paddlex.inference.utils")
        fake_base_predictor_class = type("FakeBasePredictor", (), {})
        fake_base_predictor_module = type(
            "BasePredictorModule",
            (),
            {"BasePredictor": fake_base_predictor_class},
        )()
        fake_base_pipeline_class = type("FakeBasePipeline", (), {})
        fake_base_pipeline_module = type(
            "BasePipelineModule",
            (),
            {"BasePipeline": fake_base_pipeline_class},
        )()
        parse_config_mock = Mock(return_value={"pipeline_name": "OCR"})
        fake_config_module = type("ConfigModule", (), {"parse_config": parse_config_mock})()
        resolved_path = (self.root / "release" / "paddlex" / "configs" / "pipelines" / "Custom.yaml").resolve()

        package_map = {
            "paddlex": fake_paddlex,
            "paddlex.inference": fake_inference,
            "paddlex.inference.models": fake_models,
            "paddlex.inference.models.base": fake_models_base,
            "paddlex.inference.pipelines": fake_pipelines,
            "paddlex.inference.utils": fake_utils,
        }

        def fake_ensure_virtual_package(package_name: str):
            return package_map[package_name]

        def fake_load_module_without_package_init(module_name: str):
            if module_name == "paddlex.inference.models.base.predictor.base_predictor":
                return fake_base_predictor_module
            if module_name == "paddlex.inference.pipelines.base":
                return fake_base_pipeline_module
            if module_name == "paddlex.utils.config":
                return fake_config_module
            raise AssertionError(f"Unexpected module load: {module_name}")

        load_module_mock = Mock(side_effect=fake_load_module_without_package_init)

        with patch.dict(module.sys.modules, {"paddlex.inference.utils.pp_option": None}, clear=False), \
             patch.object(module, "_ensure_virtual_package", side_effect=fake_ensure_virtual_package), \
             patch.object(module, "_load_module_without_package_init", load_module_mock), \
             patch.object(module, "_resolve_pipeline_config_path", return_value=resolved_path) as resolve_mock:
            module._ensure_paddlex_shim()
            config = fake_inference.load_pipeline_config("Custom.yaml")

        self.assertEqual(config, {"pipeline_name": "OCR"})
        self.assertIs(fake_inference.PaddlePredictorOption, module._LocalPaddlePredictorOption)
        resolve_mock.assert_called_once_with("Custom.yaml")
        parse_config_mock.assert_called_once_with(str(resolved_path))
        self.assertNotIn(
            "paddlex.inference.utils.pp_option",
            [call.args[0] for call in load_module_mock.call_args_list],
        )

    def test_ensure_paddlex_shim_create_predictor_preloads_runtime_modules_and_forwards_kwargs(self):
        fake_paddlex = module.types.ModuleType("paddlex")
        fake_inference = module.types.ModuleType("paddlex.inference")
        fake_models = module.types.ModuleType("paddlex.inference.models")
        fake_models_base = module.types.ModuleType("paddlex.inference.models.base")
        fake_pipelines = module.types.ModuleType("paddlex.inference.pipelines")
        fake_utils = module.types.ModuleType("paddlex.inference.utils")
        predictor_instance = object()
        predictor_class_mock = Mock(return_value=predictor_instance)
        base_predictor_get_mock = Mock(return_value=predictor_class_mock)
        fake_base_predictor_class = type("FakeBasePredictor", (), {"get": base_predictor_get_mock})
        fake_base_predictor_module = type(
            "BasePredictorModule",
            (),
            {"BasePredictor": fake_base_predictor_class},
        )()
        fake_base_pipeline_class = type("FakeBasePipeline", (), {})
        fake_base_pipeline_module = type(
            "BasePipelineModule",
            (),
            {"BasePipeline": fake_base_pipeline_class},
        )()

        package_map = {
            "paddlex": fake_paddlex,
            "paddlex.inference": fake_inference,
            "paddlex.inference.models": fake_models,
            "paddlex.inference.models.base": fake_models_base,
            "paddlex.inference.pipelines": fake_pipelines,
            "paddlex.inference.utils": fake_utils,
        }
        runtime_module_names = [
            "paddlex.inference.models.text_detection.predictor",
            "paddlex.inference.models.text_recognition.predictor",
        ]

        def fake_ensure_virtual_package(package_name: str):
            return package_map[package_name]

        def fake_load_module_without_package_init(module_name: str):
            if module_name == "paddlex.inference.models.base.predictor.base_predictor":
                return fake_base_predictor_module
            if module_name == "paddlex.inference.pipelines.base":
                return fake_base_pipeline_module
            if module_name in runtime_module_names:
                return object()
            raise AssertionError(f"Unexpected module load: {module_name}")

        load_module_mock = Mock(side_effect=fake_load_module_without_package_init)
        model_dir = self.root / "models" / "ocr"
        pp_option = object()
        hpi_config = {"profile": "balanced"}
        genai_config = {"enabled": True}

        with patch.dict(module.sys.modules, {"paddlex.inference.utils.pp_option": None}, clear=False), \
             patch.object(module, "_ensure_virtual_package", side_effect=fake_ensure_virtual_package), \
             patch.object(module, "_load_module_without_package_init", load_module_mock):
            module._ensure_paddlex_shim()
            load_module_mock.reset_mock()

            result = fake_inference.create_predictor(
                "TextRecognition",
                model_dir=model_dir,
                device="gpu:0",
                pp_option=pp_option,
                use_hpip=True,
                hpi_config=hpi_config,
                genai_config=genai_config,
                extra_flag="enabled",
            )

        self.assertIs(result, predictor_instance)
        base_predictor_get_mock.assert_called_once_with("TextRecognition")
        self.assertEqual(
            [call.args[0] for call in load_module_mock.call_args_list],
            runtime_module_names,
        )
        predictor_class_mock.assert_called_once_with(
            model_dir=str(model_dir),
            config=None,
            device="gpu:0",
            pp_option=pp_option,
            use_hpip=True,
            hpi_config=hpi_config,
            genai_config=genai_config,
            model_name="TextRecognition",
            extra_flag="enabled",
        )

    def test_ensure_paddlex_shim_create_predictor_forwards_default_runtime_kwargs_without_model_dir(self):
        fake_paddlex = module.types.ModuleType("paddlex")
        fake_inference = module.types.ModuleType("paddlex.inference")
        fake_models = module.types.ModuleType("paddlex.inference.models")
        fake_models_base = module.types.ModuleType("paddlex.inference.models.base")
        fake_pipelines = module.types.ModuleType("paddlex.inference.pipelines")
        fake_utils = module.types.ModuleType("paddlex.inference.utils")
        predictor_instance = object()
        predictor_class_mock = Mock(return_value=predictor_instance)
        base_predictor_get_mock = Mock(return_value=predictor_class_mock)
        fake_base_predictor_class = type("FakeBasePredictor", (), {"get": base_predictor_get_mock})
        fake_base_predictor_module = type(
            "BasePredictorModule",
            (),
            {"BasePredictor": fake_base_predictor_class},
        )()
        fake_base_pipeline_class = type("FakeBasePipeline", (), {})
        fake_base_pipeline_module = type(
            "BasePipelineModule",
            (),
            {"BasePipeline": fake_base_pipeline_class},
        )()

        package_map = {
            "paddlex": fake_paddlex,
            "paddlex.inference": fake_inference,
            "paddlex.inference.models": fake_models,
            "paddlex.inference.models.base": fake_models_base,
            "paddlex.inference.pipelines": fake_pipelines,
            "paddlex.inference.utils": fake_utils,
        }
        runtime_module_names = [
            "paddlex.inference.models.text_detection.predictor",
            "paddlex.inference.models.text_recognition.predictor",
        ]

        def fake_ensure_virtual_package(package_name: str):
            return package_map[package_name]

        def fake_load_module_without_package_init(module_name: str):
            if module_name == "paddlex.inference.models.base.predictor.base_predictor":
                return fake_base_predictor_module
            if module_name == "paddlex.inference.pipelines.base":
                return fake_base_pipeline_module
            if module_name in runtime_module_names:
                return object()
            raise AssertionError(f"Unexpected module load: {module_name}")

        load_module_mock = Mock(side_effect=fake_load_module_without_package_init)

        with patch.dict(module.sys.modules, {"paddlex.inference.utils.pp_option": None}, clear=False), \
             patch.object(module, "_ensure_virtual_package", side_effect=fake_ensure_virtual_package), \
             patch.object(module, "_load_module_without_package_init", load_module_mock):
            module._ensure_paddlex_shim()
            load_module_mock.reset_mock()

            result = fake_inference.create_predictor("TextDetection")

        self.assertIs(result, predictor_instance)
        base_predictor_get_mock.assert_called_once_with("TextDetection")
        self.assertEqual(
            [call.args[0] for call in load_module_mock.call_args_list],
            runtime_module_names,
        )
        predictor_class_mock.assert_called_once_with(
            model_dir=None,
            config=None,
            device=None,
            pp_option=None,
            use_hpip=False,
            hpi_config=None,
            genai_config=None,
            model_name="TextDetection",
        )

    def test_ensure_paddlex_shim_create_pipeline_rejects_unsupported_pipeline_config(self):
        fake_paddlex = module.types.ModuleType("paddlex")
        fake_inference = module.types.ModuleType("paddlex.inference")
        fake_models = module.types.ModuleType("paddlex.inference.models")
        fake_models_base = module.types.ModuleType("paddlex.inference.models.base")
        fake_pipelines = module.types.ModuleType("paddlex.inference.pipelines")
        fake_utils = module.types.ModuleType("paddlex.inference.utils")
        fake_base_predictor_class = type("FakeBasePredictor", (), {})
        fake_base_predictor_module = type(
            "BasePredictorModule",
            (),
            {"BasePredictor": fake_base_predictor_class},
        )()
        fake_base_pipeline_class = type("FakeBasePipeline", (), {})
        fake_base_pipeline_module = type(
            "BasePipelineModule",
            (),
            {"BasePipeline": fake_base_pipeline_class},
        )()

        package_map = {
            "paddlex": fake_paddlex,
            "paddlex.inference": fake_inference,
            "paddlex.inference.models": fake_models,
            "paddlex.inference.models.base": fake_models_base,
            "paddlex.inference.pipelines": fake_pipelines,
            "paddlex.inference.utils": fake_utils,
        }

        def fake_ensure_virtual_package(package_name: str):
            return package_map[package_name]

        def fake_load_module_without_package_init(module_name: str):
            if module_name == "paddlex.inference.models.base.predictor.base_predictor":
                return fake_base_predictor_module
            if module_name == "paddlex.inference.pipelines.base":
                return fake_base_pipeline_module
            raise AssertionError(f"Unexpected module load: {module_name}")

        load_module_mock = Mock(side_effect=fake_load_module_without_package_init)
        config = {"pipeline_name": "Layout", "use_hpip": True}

        with patch.dict(module.sys.modules, {"paddlex.inference.utils.pp_option": None}, clear=False), \
             patch.object(module, "_ensure_virtual_package", side_effect=fake_ensure_virtual_package), \
             patch.object(module, "_load_module_without_package_init", load_module_mock), \
             patch.object(module, "_ensure_paddlex_ocr_components_shim") as ensure_components_mock:
            module._ensure_paddlex_shim()

            with self.assertRaisesRegex(
                ValueError,
                "Unsupported PaddleX pipeline for offline OCR loader: Layout",
            ):
                fake_inference.create_pipeline(config=config)

        self.assertEqual(config, {"pipeline_name": "Layout", "use_hpip": True})
        ensure_components_mock.assert_not_called()
        self.assertEqual(
            [call.args[0] for call in load_module_mock.call_args_list],
            [
                "paddlex.inference.models.base.predictor.base_predictor",
                "paddlex.inference.pipelines.base",
            ],
        )

    def test_ensure_paddlex_shim_create_pipeline_rejects_unsupported_config_without_loading_runtime_modules(self):
        fake_paddlex = module.types.ModuleType("paddlex")
        fake_inference = module.types.ModuleType("paddlex.inference")
        fake_models = module.types.ModuleType("paddlex.inference.models")
        fake_models_base = module.types.ModuleType("paddlex.inference.models.base")
        fake_pipelines = module.types.ModuleType("paddlex.inference.pipelines")
        fake_utils = module.types.ModuleType("paddlex.inference.utils")
        fake_base_predictor_class = type("FakeBasePredictor", (), {})
        fake_base_predictor_module = type(
            "BasePredictorModule",
            (),
            {"BasePredictor": fake_base_predictor_class},
        )()
        fake_base_pipeline_class = type("FakeBasePipeline", (), {})
        fake_base_pipeline_module = type(
            "BasePipelineModule",
            (),
            {"BasePipeline": fake_base_pipeline_class},
        )()

        package_map = {
            "paddlex": fake_paddlex,
            "paddlex.inference": fake_inference,
            "paddlex.inference.models": fake_models,
            "paddlex.inference.models.base": fake_models_base,
            "paddlex.inference.pipelines": fake_pipelines,
            "paddlex.inference.utils": fake_utils,
        }

        def fake_ensure_virtual_package(package_name: str):
            return package_map[package_name]

        def fake_load_module_without_package_init(module_name: str):
            if module_name == "paddlex.inference.models.base.predictor.base_predictor":
                return fake_base_predictor_module
            if module_name == "paddlex.inference.pipelines.base":
                return fake_base_pipeline_module
            raise AssertionError(f"Unexpected module load: {module_name}")

        load_module_mock = Mock(side_effect=fake_load_module_without_package_init)

        with patch.dict(module.sys.modules, {"paddlex.inference.utils.pp_option": None}, clear=False),              patch.object(module, "_ensure_virtual_package", side_effect=fake_ensure_virtual_package),              patch.object(module, "_load_module_without_package_init", load_module_mock),              patch.object(module, "_ensure_paddlex_ocr_components_shim") as ensure_components_mock:
            module._ensure_paddlex_shim()
            load_module_mock.reset_mock()

            with self.assertRaisesRegex(ValueError, "Unsupported PaddleX pipeline for offline OCR loader: DocLayout"):
                fake_inference.create_pipeline(config={"pipeline_name": "DocLayout"})

        ensure_components_mock.assert_not_called()
        load_module_mock.assert_not_called()

    def test_ensure_paddlex_shim_create_pipeline_rejects_missing_pipeline_and_config_without_loading_runtime_modules(self):
        fake_paddlex = module.types.ModuleType("paddlex")
        fake_inference = module.types.ModuleType("paddlex.inference")
        fake_models = module.types.ModuleType("paddlex.inference.models")
        fake_models_base = module.types.ModuleType("paddlex.inference.models.base")
        fake_pipelines = module.types.ModuleType("paddlex.inference.pipelines")
        fake_utils = module.types.ModuleType("paddlex.inference.utils")
        fake_base_predictor_class = type("FakeBasePredictor", (), {})
        fake_base_predictor_module = type(
            "BasePredictorModule",
            (),
            {"BasePredictor": fake_base_predictor_class},
        )()
        fake_base_pipeline_class = type("FakeBasePipeline", (), {})
        fake_base_pipeline_module = type(
            "BasePipelineModule",
            (),
            {"BasePipeline": fake_base_pipeline_class},
        )()

        package_map = {
            "paddlex": fake_paddlex,
            "paddlex.inference": fake_inference,
            "paddlex.inference.models": fake_models,
            "paddlex.inference.models.base": fake_models_base,
            "paddlex.inference.pipelines": fake_pipelines,
            "paddlex.inference.utils": fake_utils,
        }

        def fake_ensure_virtual_package(package_name: str):
            return package_map[package_name]

        def fake_load_module_without_package_init(module_name: str):
            if module_name == "paddlex.inference.models.base.predictor.base_predictor":
                return fake_base_predictor_module
            if module_name == "paddlex.inference.pipelines.base":
                return fake_base_pipeline_module
            raise AssertionError(f"Unexpected module load: {module_name}")

        load_module_mock = Mock(side_effect=fake_load_module_without_package_init)

        with patch.dict(module.sys.modules, {"paddlex.inference.utils.pp_option": None}, clear=False),              patch.object(module, "_ensure_virtual_package", side_effect=fake_ensure_virtual_package),              patch.object(module, "_load_module_without_package_init", load_module_mock),              patch.object(module, "_ensure_paddlex_ocr_components_shim") as ensure_components_mock:
            module._ensure_paddlex_shim()
            load_module_mock.reset_mock()

            with self.assertRaisesRegex(
                ValueError,
                "Both `pipeline` and `config` cannot be None at the same time.",
            ):
                fake_inference.create_pipeline()

        ensure_components_mock.assert_not_called()
        load_module_mock.assert_not_called()

    def test_ensure_paddlex_shim_create_pipeline_uses_copied_ocr_config_and_forwards_runtime_kwargs(self):
        fake_paddlex = module.types.ModuleType("paddlex")
        fake_inference = module.types.ModuleType("paddlex.inference")
        fake_models = module.types.ModuleType("paddlex.inference.models")
        fake_models_base = module.types.ModuleType("paddlex.inference.models.base")
        fake_pipelines = module.types.ModuleType("paddlex.inference.pipelines")
        fake_utils = module.types.ModuleType("paddlex.inference.utils")
        expected_pipeline = object()
        fake_pipeline_class = Mock(return_value=expected_pipeline)
        base_pipeline_get_mock = Mock(return_value=fake_pipeline_class)
        fake_base_predictor_class = type("FakeBasePredictor", (), {})
        fake_base_predictor_module = type(
            "BasePredictorModule",
            (),
            {"BasePredictor": fake_base_predictor_class},
        )()
        fake_base_pipeline_class = type("FakeBasePipeline", (), {"get": base_pipeline_get_mock})
        fake_base_pipeline_module = type(
            "BasePipelineModule",
            (),
            {"BasePipeline": fake_base_pipeline_class},
        )()

        package_map = {
            "paddlex": fake_paddlex,
            "paddlex.inference": fake_inference,
            "paddlex.inference.models": fake_models,
            "paddlex.inference.models.base": fake_models_base,
            "paddlex.inference.pipelines": fake_pipelines,
            "paddlex.inference.utils": fake_utils,
        }

        def fake_ensure_virtual_package(package_name: str):
            return package_map[package_name]

        def fake_load_module_without_package_init(module_name: str):
            if module_name == "paddlex.inference.models.base.predictor.base_predictor":
                return fake_base_predictor_module
            if module_name == "paddlex.inference.pipelines.base":
                return fake_base_pipeline_module
            if module_name == "paddlex.inference.pipelines.ocr.pipeline":
                return object()
            raise AssertionError(f"Unexpected module load: {module_name}")

        load_module_mock = Mock(side_effect=fake_load_module_without_package_init)
        pp_option = object()
        hpi_config = {"backend": "gpu"}
        config = {
            "pipeline_name": "OCR",
            "use_hpip": True,
            "hpi_config": hpi_config,
            "preserved": "value",
        }

        with patch.dict(module.sys.modules, {"paddlex.inference.utils.pp_option": None}, clear=False), \
             patch.object(module, "_ensure_virtual_package", side_effect=fake_ensure_virtual_package), \
             patch.object(module, "_load_module_without_package_init", load_module_mock), \
             patch.object(module, "_ensure_paddlex_ocr_components_shim") as ensure_components_mock:
            module._ensure_paddlex_shim()
            load_module_mock.reset_mock()

            result = fake_inference.create_pipeline(
                config=config,
                device="gpu:0",
                pp_option=pp_option,
                extra_flag="enabled",
            )

        self.assertIs(result, expected_pipeline)
        self.assertEqual(
            config,
            {
                "pipeline_name": "OCR",
                "use_hpip": True,
                "hpi_config": hpi_config,
                "preserved": "value",
            },
        )
        ensure_components_mock.assert_called_once_with()
        load_module_mock.assert_called_once_with("paddlex.inference.pipelines.ocr.pipeline")
        base_pipeline_get_mock.assert_called_once_with("OCR")
        self.assertIsNot(fake_pipeline_class.call_args.kwargs["config"], config)
        fake_pipeline_class.assert_called_once_with(
            config={"pipeline_name": "OCR", "preserved": "value"},
            device="gpu:0",
            pp_option=pp_option,
            use_hpip=True,
            hpi_config=hpi_config,
            extra_flag="enabled",
        )

    def test_ensure_paddlex_shim_create_pipeline_loads_named_ocr_pipeline_without_explicit_config(self):
        fake_paddlex = module.types.ModuleType("paddlex")
        fake_inference = module.types.ModuleType("paddlex.inference")
        fake_models = module.types.ModuleType("paddlex.inference.models")
        fake_models_base = module.types.ModuleType("paddlex.inference.models.base")
        fake_pipelines = module.types.ModuleType("paddlex.inference.pipelines")
        fake_utils = module.types.ModuleType("paddlex.inference.utils")
        expected_pipeline = object()
        fake_pipeline_class = Mock(return_value=expected_pipeline)
        base_pipeline_get_mock = Mock(return_value=fake_pipeline_class)
        fake_base_predictor_class = type("FakeBasePredictor", (), {})
        fake_base_predictor_module = type(
            "BasePredictorModule",
            (),
            {"BasePredictor": fake_base_predictor_class},
        )()
        fake_base_pipeline_class = type("FakeBasePipeline", (), {"get": base_pipeline_get_mock})
        fake_base_pipeline_module = type(
            "BasePipelineModule",
            (),
            {"BasePipeline": fake_base_pipeline_class},
        )()

        package_map = {
            "paddlex": fake_paddlex,
            "paddlex.inference": fake_inference,
            "paddlex.inference.models": fake_models,
            "paddlex.inference.models.base": fake_models_base,
            "paddlex.inference.pipelines": fake_pipelines,
            "paddlex.inference.utils": fake_utils,
        }

        def fake_ensure_virtual_package(package_name: str):
            return package_map[package_name]

        def fake_load_module_without_package_init(module_name: str):
            if module_name == "paddlex.inference.models.base.predictor.base_predictor":
                return fake_base_predictor_module
            if module_name == "paddlex.inference.pipelines.base":
                return fake_base_pipeline_module
            if module_name == "paddlex.inference.pipelines.ocr.pipeline":
                return object()
            raise AssertionError(f"Unexpected module load: {module_name}")

        load_module_mock = Mock(side_effect=fake_load_module_without_package_init)
        pp_option = object()
        minimal_config = {"pipeline_name": "OCR", "preserved": "value"}

        with patch.dict(module.sys.modules, {"paddlex.inference.utils.pp_option": None}, clear=False), \
             patch.object(module, "_ensure_virtual_package", side_effect=fake_ensure_virtual_package), \
             patch.object(module, "_load_module_without_package_init", load_module_mock), \
             patch.object(module, "_ensure_paddlex_ocr_components_shim") as ensure_components_mock, \
             patch.object(module, "_get_minimal_ocr_pipeline_config", return_value=minimal_config) as minimal_config_mock:
            module._ensure_paddlex_shim()
            load_module_mock.reset_mock()

            result = fake_inference.create_pipeline(
                pipeline="OCR",
                device="gpu:0",
                pp_option=pp_option,
                extra_flag="enabled",
            )

        self.assertIs(result, expected_pipeline)
        minimal_config_mock.assert_called_once_with()
        ensure_components_mock.assert_called_once_with()
        load_module_mock.assert_called_once_with("paddlex.inference.pipelines.ocr.pipeline")
        base_pipeline_get_mock.assert_called_once_with("OCR")
        fake_pipeline_class.assert_called_once_with(
            config={"pipeline_name": "OCR", "preserved": "value"},
            device="gpu:0",
            pp_option=pp_option,
            use_hpip=False,
            hpi_config=None,
            extra_flag="enabled",
        )

    def test_ensure_paddlex_shim_create_pipeline_explicit_runtime_kwargs_override_config_values(self):
        fake_paddlex = module.types.ModuleType("paddlex")
        fake_inference = module.types.ModuleType("paddlex.inference")
        fake_models = module.types.ModuleType("paddlex.inference.models")
        fake_models_base = module.types.ModuleType("paddlex.inference.models.base")
        fake_pipelines = module.types.ModuleType("paddlex.inference.pipelines")
        fake_utils = module.types.ModuleType("paddlex.inference.utils")
        expected_pipeline = object()
        fake_pipeline_class = Mock(return_value=expected_pipeline)
        base_pipeline_get_mock = Mock(return_value=fake_pipeline_class)
        fake_base_predictor_class = type("FakeBasePredictor", (), {})
        fake_base_predictor_module = type(
            "BasePredictorModule",
            (),
            {"BasePredictor": fake_base_predictor_class},
        )()
        fake_base_pipeline_class = type("FakeBasePipeline", (), {"get": base_pipeline_get_mock})
        fake_base_pipeline_module = type(
            "BasePipelineModule",
            (),
            {"BasePipeline": fake_base_pipeline_class},
        )()

        package_map = {
            "paddlex": fake_paddlex,
            "paddlex.inference": fake_inference,
            "paddlex.inference.models": fake_models,
            "paddlex.inference.models.base": fake_models_base,
            "paddlex.inference.pipelines": fake_pipelines,
            "paddlex.inference.utils": fake_utils,
        }

        def fake_ensure_virtual_package(package_name: str):
            return package_map[package_name]

        def fake_load_module_without_package_init(module_name: str):
            if module_name == "paddlex.inference.models.base.predictor.base_predictor":
                return fake_base_predictor_module
            if module_name == "paddlex.inference.pipelines.base":
                return fake_base_pipeline_module
            if module_name == "paddlex.inference.pipelines.ocr.pipeline":
                return object()
            raise AssertionError(f"Unexpected module load: {module_name}")

        load_module_mock = Mock(side_effect=fake_load_module_without_package_init)
        pp_option = object()
        config_hpi_config = {"backend": "gpu"}
        explicit_hpi_config = {"backend": "cpu"}
        config = {
            "pipeline_name": "OCR",
            "use_hpip": True,
            "hpi_config": config_hpi_config,
            "preserved": "value",
        }

        with patch.dict(module.sys.modules, {"paddlex.inference.utils.pp_option": None}, clear=False), \
             patch.object(module, "_ensure_virtual_package", side_effect=fake_ensure_virtual_package), \
             patch.object(module, "_load_module_without_package_init", load_module_mock), \
             patch.object(module, "_ensure_paddlex_ocr_components_shim") as ensure_components_mock:
            module._ensure_paddlex_shim()
            load_module_mock.reset_mock()

            result = fake_inference.create_pipeline(
                config=config,
                device="cpu",
                pp_option=pp_option,
                use_hpip=False,
                hpi_config=explicit_hpi_config,
                extra_flag="enabled",
            )

        self.assertIs(result, expected_pipeline)
        self.assertEqual(
            config,
            {
                "pipeline_name": "OCR",
                "use_hpip": True,
                "hpi_config": config_hpi_config,
                "preserved": "value",
            },
        )
        ensure_components_mock.assert_called_once_with()
        load_module_mock.assert_called_once_with("paddlex.inference.pipelines.ocr.pipeline")
        base_pipeline_get_mock.assert_called_once_with("OCR")
        self.assertIsNot(fake_pipeline_class.call_args.kwargs["config"], config)
        fake_pipeline_class.assert_called_once_with(
            config={"pipeline_name": "OCR", "preserved": "value"},
            device="cpu",
            pp_option=pp_option,
            use_hpip=False,
            hpi_config=explicit_hpi_config,
            extra_flag="enabled",
        )

    def test_local_paddle_predictor_option_set_device_ignores_empty_value(self):
        option = module._LocalPaddlePredictorOption()
        option.device_type = "gpu"
        option.device_id = 7
        original_cfg = dict(option._cfg)

        option.set_device("")

        self.assertEqual(option._cfg, original_cfg)
        self.assertEqual(option.device_type, "gpu")
        self.assertEqual(option.device_id, 7)

    def test_local_paddle_predictor_option_init_ignores_empty_device_value(self):
        option = module._LocalPaddlePredictorOption(device="")

        self.assertEqual(option._cfg, {})
        self.assertIsNone(option.device_type)
        self.assertIsNone(option.device_id)

    def test_local_paddle_predictor_option_set_device_sets_cpu_and_clears_device_id(self):
        option = module._LocalPaddlePredictorOption()
        option.device_type = "gpu"
        option.device_id = 7

        option.set_device("cpu")

        self.assertEqual(option.device_type, "cpu")
        self.assertIsNone(option.device_id)
        self.assertEqual(option._cfg, {"device_type": "cpu", "device_id": None})

    def test_local_paddle_predictor_option_set_device_sets_gpu_without_device_id_for_trailing_colon(self):
        option = module._LocalPaddlePredictorOption(device_type="cpu", device_id=7)

        option.set_device("gpu:")

        self.assertEqual(option.device_type, "gpu")
        self.assertIsNone(option.device_id)
        self.assertEqual(option._cfg, {"device_type": "gpu", "device_id": None})

    def test_local_paddle_predictor_option_set_device_rejects_empty_device_id_list(self):
        option = module._LocalPaddlePredictorOption(device_type="gpu", device_id=7)
        original_cfg = dict(option._cfg)

        with self.assertRaisesRegex(ValueError, "Invalid device: gpu:,"):
            option.set_device("gpu:,")

        self.assertEqual(option._cfg, original_cfg)
        self.assertEqual(option.device_type, "gpu")
        self.assertEqual(option.device_id, 7)

    def test_local_paddle_predictor_option_set_device_rejects_cpu_with_explicit_device_id(self):
        option = module._LocalPaddlePredictorOption()

        with self.assertRaisesRegex(ValueError, "No Device ID should be specified for CPUs"):
            option.set_device("cpu:0")

        self.assertIsNone(option.device_type)
        self.assertIsNone(option.device_id)

    def test_local_paddle_predictor_option_set_device_uses_first_gpu_device_id(self):
        option = module._LocalPaddlePredictorOption()

        option.set_device("gpu:3,7")

        self.assertEqual(option.device_type, "gpu")
        self.assertEqual(option.device_id, 3)

    def test_local_paddle_predictor_option_set_device_normalizes_whitespace_and_device_type_case(self):
        option = module._LocalPaddlePredictorOption()

        option.set_device(" GPU : 3 , 7 ")

        self.assertEqual(option.device_type, "gpu")
        self.assertEqual(option.device_id, 3)
        self.assertEqual(option._cfg, {"device_type": "gpu", "device_id": 3})

    def test_local_paddle_predictor_option_set_device_rejects_non_numeric_device_id(self):
        option = module._LocalPaddlePredictorOption()

        with self.assertRaisesRegex(ValueError, "Invalid device: gpu:abc"):
            option.set_device("gpu:abc")

        self.assertIsNone(option.device_type)
        self.assertIsNone(option.device_id)

    def test_local_paddle_predictor_option_set_device_rejects_unsupported_device_type(self):
        option = module._LocalPaddlePredictorOption()

        with self.assertRaisesRegex(ValueError, "received 'tpu'"):
            option.set_device("tpu")

        self.assertIsNone(option.device_type)
        self.assertIsNone(option.device_id)

    def test_local_paddle_predictor_option_device_type_setter_updates_cfg(self):
        option = module._LocalPaddlePredictorOption()

        option.device_type = "gpu"

        self.assertEqual(option.device_type, "gpu")
        self.assertEqual(option._cfg, {"device_type": "gpu"})

    def test_local_paddle_predictor_option_device_type_setter_rejects_unsupported_device_type(self):
        option = module._LocalPaddlePredictorOption(device_type="gpu", device_id=7)
        original_cfg = dict(option._cfg)

        with self.assertRaisesRegex(ValueError, "received 'tpu'"):
            option.device_type = "tpu"

        self.assertEqual(option._cfg, original_cfg)
        self.assertEqual(option.device_type, "gpu")
        self.assertEqual(option.device_id, 7)

    def test_local_paddle_predictor_option_device_id_setter_updates_cfg_without_affecting_device_type(self):
        option = module._LocalPaddlePredictorOption(device_type="gpu")

        option.device_id = 7

        self.assertEqual(option.device_type, "gpu")
        self.assertEqual(option.device_id, 7)
        self.assertEqual(option._cfg, {"device_type": "gpu", "device_id": 7})

    def test_local_paddle_predictor_option_cpu_threads_setter_updates_cfg(self):
        option = module._LocalPaddlePredictorOption()

        option.cpu_threads = 6

        self.assertEqual(option.cpu_threads, 6)
        self.assertEqual(option._cfg, {"cpu_threads": 6})

    def test_local_paddle_predictor_option_trt_shape_range_info_path_setter_updates_alias_view(self):
        option = module._LocalPaddlePredictorOption()

        option.trt_shape_range_info_path = "cached.pbtxt"

        self.assertEqual(option.trt_shape_range_info_path, "cached.pbtxt")
        self.assertEqual(option.shape_info_filename, "cached.pbtxt")
        self.assertEqual(option._cfg, {"trt_shape_range_info_path": "cached.pbtxt"})

    def test_local_paddle_predictor_option_enable_new_ir_setter_updates_cfg(self):
        option = module._LocalPaddlePredictorOption()

        option.enable_new_ir = False

        self.assertFalse(option.enable_new_ir)
        self.assertEqual(option._cfg, {"enable_new_ir": False})

    def test_local_paddle_predictor_option_enable_cinn_setter_updates_cfg(self):
        option = module._LocalPaddlePredictorOption()

        option.enable_cinn = False

        self.assertFalse(option.enable_cinn)
        self.assertEqual(option._cfg, {"enable_cinn": False})

    def test_local_paddle_predictor_option_trt_use_dynamic_shapes_setter_updates_cfg(self):
        option = module._LocalPaddlePredictorOption()

        option.trt_use_dynamic_shapes = False

        self.assertFalse(option.trt_use_dynamic_shapes)
        self.assertEqual(option._cfg, {"trt_use_dynamic_shapes": False})

    def test_local_paddle_predictor_option_trt_collect_shape_range_info_setter_updates_cfg(self):
        option = module._LocalPaddlePredictorOption()

        option.trt_collect_shape_range_info = False

        self.assertFalse(option.trt_collect_shape_range_info)
        self.assertEqual(option._cfg, {"trt_collect_shape_range_info": False})

    def test_local_paddle_predictor_option_trt_discard_cached_shape_range_info_setter_updates_cfg(self):
        option = module._LocalPaddlePredictorOption()

        option.trt_discard_cached_shape_range_info = False

        self.assertFalse(option.trt_discard_cached_shape_range_info)
        self.assertEqual(option._cfg, {"trt_discard_cached_shape_range_info": False})

    def test_local_paddle_predictor_option_trt_allow_rebuild_at_runtime_setter_updates_cfg(self):
        option = module._LocalPaddlePredictorOption()

        option.trt_allow_rebuild_at_runtime = False

        self.assertFalse(option.trt_allow_rebuild_at_runtime)
        self.assertEqual(option._cfg, {"trt_allow_rebuild_at_runtime": False})

    def test_local_paddle_predictor_option_mkldnn_cache_capacity_setter_updates_cfg(self):
        option = module._LocalPaddlePredictorOption()

        option.mkldnn_cache_capacity = 24

        self.assertEqual(option.mkldnn_cache_capacity, 24)
        self.assertEqual(option._cfg, {"mkldnn_cache_capacity": 24})

    def test_local_paddle_predictor_option_trt_dynamic_shape_input_data_setter_updates_cfg(self):
        option = module._LocalPaddlePredictorOption()
        input_data = {"x": [1, 3, 32, 32]}

        option.trt_dynamic_shape_input_data = input_data

        self.assertEqual(option.trt_dynamic_shape_input_data, input_data)
        self.assertEqual(option._cfg, {"trt_dynamic_shape_input_data": input_data})

    def test_local_paddle_predictor_option_delete_pass_setter_updates_cfg(self):
        option = module._LocalPaddlePredictorOption()

        option.delete_pass = ["skip_a", "skip_b"]

        self.assertEqual(option.delete_pass, ["skip_a", "skip_b"])
        self.assertEqual(option._cfg, {"delete_pass": ["skip_a", "skip_b"]})

    def test_local_paddle_predictor_option_shape_info_filename_aliases_trt_shape_range_info_path(self):
        option = module._LocalPaddlePredictorOption()

        option.shape_info_filename = "shape-info.pbtxt"

        self.assertEqual(option.shape_info_filename, "shape-info.pbtxt")
        self.assertEqual(option.trt_shape_range_info_path, "shape-info.pbtxt")
        self.assertEqual(option._cfg, {"trt_shape_range_info_path": "shape-info.pbtxt"})

    def test_local_paddle_predictor_option_shape_info_filename_reads_underlying_trt_shape_range_info_path(self):
        option = module._LocalPaddlePredictorOption(trt_shape_range_info_path="cached.pbtxt")

        self.assertEqual(option.shape_info_filename, "cached.pbtxt")
        self.assertEqual(option.trt_shape_range_info_path, "cached.pbtxt")
        self.assertEqual(option._cfg, {"trt_shape_range_info_path": "cached.pbtxt"})

    def test_local_paddle_predictor_option_init_accepts_shape_info_filename_alias(self):
        option = module._LocalPaddlePredictorOption(shape_info_filename="shape-info.pbtxt")

        self.assertEqual(option.shape_info_filename, "shape-info.pbtxt")
        self.assertEqual(option.trt_shape_range_info_path, "shape-info.pbtxt")
        self.assertEqual(option._cfg, {"trt_shape_range_info_path": "shape-info.pbtxt"})

    def test_local_paddle_predictor_option_copy_deep_copies_nested_config(self):
        option = module._LocalPaddlePredictorOption(trt_cfg_setting={"shape": {"min": [1]}})

        copied = option.copy()
        copied.trt_cfg_setting["shape"]["min"].append(2)

        self.assertIsNot(copied, option)
        self.assertEqual(option.trt_cfg_setting, {"shape": {"min": [1]}})
        self.assertEqual(copied.trt_cfg_setting, {"shape": {"min": [1, 2]}})

    def test_local_paddle_predictor_option_support_accessors_return_class_constants(self):
        option = module._LocalPaddlePredictorOption()

        self.assertIs(option.get_support_run_mode(), module._LocalPaddlePredictorOption.SUPPORT_RUN_MODE)
        self.assertIs(option.get_support_device(), module._LocalPaddlePredictorOption.SUPPORT_DEVICE)

    def test_local_paddle_predictor_option_run_mode_setter_updates_cfg(self):
        option = module._LocalPaddlePredictorOption()

        option.run_mode = "trt_fp16"

        self.assertEqual(option.run_mode, "trt_fp16")
        self.assertEqual(option._cfg, {"run_mode": "trt_fp16"})

    def test_local_paddle_predictor_option_run_mode_rejects_unsupported_value(self):
        option = module._LocalPaddlePredictorOption()

        with self.assertRaisesRegex(ValueError, "received 'invalid_mode'"):
            option.run_mode = "invalid_mode"

        self.assertIsNone(option.run_mode)
        self.assertEqual(option._cfg, {})

    def test_local_paddle_predictor_option_trt_dynamic_shapes_setter_updates_cfg(self):
        option = module._LocalPaddlePredictorOption()
        shapes = {"x": {"min_shape": [1, 3, 32, 32]}}

        option.trt_dynamic_shapes = shapes

        self.assertEqual(option.trt_dynamic_shapes, shapes)
        self.assertEqual(option._cfg, {"trt_dynamic_shapes": shapes})

    def test_local_paddle_predictor_option_trt_dynamic_shapes_rejects_non_dict_without_mutating_state(self):
        initial_shapes = {"x": {"min_shape": [1, 3, 32, 32]}}
        option = module._LocalPaddlePredictorOption(trt_dynamic_shapes=initial_shapes)

        with self.assertRaisesRegex(AssertionError, "trt_dynamic_shapes must be a dict or None"):
            option.trt_dynamic_shapes = ["not", "a", "dict"]

        self.assertEqual(option.trt_dynamic_shapes, initial_shapes)
        self.assertEqual(option._cfg, {"trt_dynamic_shapes": initial_shapes})

    def test_local_paddle_predictor_option_trt_cfg_setting_setter_updates_cfg(self):
        option = module._LocalPaddlePredictorOption()
        config = {"shape": {"min": [1]}}

        option.trt_cfg_setting = config

        self.assertEqual(option.trt_cfg_setting, config)
        self.assertEqual(option._cfg, {"trt_cfg_setting": config})

    def test_local_paddle_predictor_option_trt_cfg_setting_rejects_non_dict_without_mutating_state(self):
        initial_config = {"shape": {"min": [1]}}
        option = module._LocalPaddlePredictorOption(trt_cfg_setting=initial_config)

        with self.assertRaisesRegex(AssertionError, r"The trt_cfg_setting must be `dict` type"):
            option.trt_cfg_setting = ["not", "a", "dict"]

        self.assertEqual(option.trt_cfg_setting, initial_config)
        self.assertEqual(option._cfg, {"trt_cfg_setting": initial_config})

    def test_local_paddle_predictor_option_eq_compares_cfg_and_rejects_other_types(self):
        option = module._LocalPaddlePredictorOption(run_mode="paddle", device="gpu:1")
        same_option = module._LocalPaddlePredictorOption(run_mode="paddle", device="gpu:1")
        different_option = module._LocalPaddlePredictorOption(run_mode="paddle_fp16", device="gpu:1")

        self.assertTrue(option == same_option)
        self.assertFalse(option == different_option)
        self.assertFalse(option == object())

    def test_local_paddle_predictor_option_str_formats_cfg_entries_in_order(self):
        option = module._LocalPaddlePredictorOption(run_mode="paddle", device="gpu:1")

        self.assertEqual(str(option), "run_mode: paddle,  device_type: gpu,  device_id: 1")

    def test_local_paddle_predictor_option_getattr_rejects_missing_key(self):
        option = module._LocalPaddlePredictorOption(run_mode="paddle")

        with self.assertRaisesRegex(Exception, r"The key \(missing_key\) is not found in cfg:"):
            _ = option.missing_key

        self.assertEqual(option._cfg, {"run_mode": "paddle"})

    def test_local_paddle_predictor_option_getattr_returns_value_for_updated_non_property_key(self):
        option = module._LocalPaddlePredictorOption(run_mode="paddle")

        option._update("custom_key", "custom-value")

        self.assertEqual(option.custom_key, "custom-value")
        self.assertEqual(option._cfg, {"run_mode": "paddle", "custom_key": "custom-value"})

    def test_local_paddle_predictor_option_init_option_dispatches_device_alias_and_setter_backed_keywords(self):
        option = module._LocalPaddlePredictorOption()

        option._init_option(device="gpu:3", run_mode="mkldnn", cpu_threads=6)

        self.assertEqual(option.device_type, "gpu")
        self.assertEqual(option.device_id, 3)
        self.assertEqual(option.run_mode, "mkldnn")
        self.assertEqual(option.cpu_threads, 6)
        self.assertNotIn("device", option._cfg)
        self.assertEqual(
            option._cfg,
            {"device_type": "gpu", "device_id": 3, "run_mode": "mkldnn", "cpu_threads": 6},
        )

    def test_local_paddle_predictor_option_init_rejects_unsupported_keyword(self):
        option = module._LocalPaddlePredictorOption(run_mode="paddle")

        with self.assertRaisesRegex(Exception, r"unsupported_key is not supported to set!") as exc_info:
            option._init_option(unsupported_key="value")

        self.assertIn("run_mode", str(exc_info.exception))
        self.assertEqual(option._cfg, {"run_mode": "paddle"})

    def test_local_paddle_predictor_option_cpu_threads_rejects_invalid_value_without_mutating_state(self):
        option = module._LocalPaddlePredictorOption(cpu_threads=4)

        with self.assertRaises(Exception):
            option.cpu_threads = 0

        self.assertEqual(option.cpu_threads, 4)
        self.assertEqual(option._cfg, {"cpu_threads": 4})

    def test_local_paddle_predictor_option_has_setter_only_for_property_backed_attributes(self):
        option = module._LocalPaddlePredictorOption()

        self.assertTrue(option._has_setter("run_mode"))
        self.assertTrue(option._has_setter("shape_info_filename"))
        self.assertFalse(option._has_setter("set_device"))
        self.assertFalse(option._has_setter("missing_attribute"))

    def test_local_paddle_predictor_option_get_settable_attributes_lists_properties_only(self):
        option = module._LocalPaddlePredictorOption()

        attributes = option._get_settable_attributes()

        self.assertIn("run_mode", attributes)
        self.assertIn("device_type", attributes)
        self.assertIn("shape_info_filename", attributes)
        self.assertIn("mkldnn_cache_capacity", attributes)
        self.assertNotIn("missing_attribute", attributes)
        self.assertNotIn("set_device", attributes)
        self.assertNotIn("copy", attributes)
        self.assertNotIn("_init_option", attributes)

    def test_local_paddle_predictor_option_get_default_config_uses_local_defaults_without_aliasing_state(self):
        option = module._LocalPaddlePredictorOption(
            device="gpu:2",
            delete_pass=["skip_pass"],
            trt_cfg_setting={"shape": {"min": [1]}},
            trt_dynamic_shapes={"x": {"min_shape": [1, 3, 32, 32]}},
            trt_dynamic_shape_input_data={"x": [1, 3, 32, 32]},
        )

        config = option._get_default_config("ocr-model")
        config["delete_pass"].append("extra_pass")
        config["trt_cfg_setting"]["shape"]["max"] = [2]
        config["trt_dynamic_shapes"]["x"]["max_shape"] = [1, 3, 64, 64]
        config["trt_dynamic_shape_input_data"]["x"].append(64)

        self.assertEqual(config["model_name"], "ocr-model")
        self.assertEqual(config["run_mode"], "paddle")
        self.assertEqual(config["device_type"], "gpu")
        self.assertEqual(config["device_id"], 2)
        self.assertEqual(config["cpu_threads"], 10)
        self.assertTrue(config["enable_new_ir"])
        self.assertFalse(config["enable_cinn"])
        self.assertTrue(config["trt_use_dynamic_shapes"])
        self.assertTrue(config["trt_collect_shape_range_info"])
        self.assertFalse(config["trt_discard_cached_shape_range_info"])
        self.assertTrue(config["trt_allow_rebuild_at_runtime"])
        self.assertEqual(config["mkldnn_cache_capacity"], 10)

        self.assertEqual(option.delete_pass, ["skip_pass"])
        self.assertEqual(option.trt_cfg_setting, {"shape": {"min": [1]}})
        self.assertEqual(option.trt_dynamic_shapes, {"x": {"min_shape": [1, 3, 32, 32]}})
        self.assertEqual(option.trt_dynamic_shape_input_data, {"x": [1, 3, 32, 32]})

    def test_local_paddle_predictor_option_get_default_config_normalizes_cpu_device_id_without_mutating_state(self):
        option = module._LocalPaddlePredictorOption(device_type="cpu", device_id=7)

        config = option._get_default_config("ocr-model")

        self.assertEqual(config["model_name"], "ocr-model")
        self.assertEqual(config["device_type"], "cpu")
        self.assertIsNone(config["device_id"])
        self.assertEqual(option.device_type, "cpu")
        self.assertEqual(option.device_id, 7)
        self.assertEqual(option._cfg, {"device_type": "cpu", "device_id": 7})

    def test_local_paddle_predictor_option_get_default_config_preserves_explicit_trt_shape_range_info_path_without_mutating_state(self):
        option = module._LocalPaddlePredictorOption(trt_shape_range_info_path="cached.pbtxt")

        config = option._get_default_config("ocr-model")

        self.assertEqual(config["model_name"], "ocr-model")
        self.assertEqual(config["trt_shape_range_info_path"], "cached.pbtxt")
        self.assertEqual(option.trt_shape_range_info_path, "cached.pbtxt")
        self.assertEqual(option._cfg, {"trt_shape_range_info_path": "cached.pbtxt"})

    def test_local_paddle_predictor_option_get_default_config_normalizes_zero_mkldnn_cache_capacity_without_mutating_state(self):
        option = module._LocalPaddlePredictorOption(mkldnn_cache_capacity=0)

        config = option._get_default_config("ocr-model")

        self.assertEqual(config["mkldnn_cache_capacity"], 10)
        self.assertEqual(option.mkldnn_cache_capacity, 0)
        self.assertEqual(option._cfg, {"mkldnn_cache_capacity": 0})

    def test_local_paddle_predictor_option_get_default_config_preserves_explicit_override_values(self):
        option = module._LocalPaddlePredictorOption(
            enable_new_ir=False,
            enable_cinn=True,
            trt_use_dynamic_shapes=False,
            trt_collect_shape_range_info=False,
            trt_discard_cached_shape_range_info=True,
            trt_allow_rebuild_at_runtime=False,
            mkldnn_cache_capacity=24,
        )

        config = option._get_default_config("ocr-model")

        self.assertFalse(config["enable_new_ir"])
        self.assertTrue(config["enable_cinn"])
        self.assertFalse(config["trt_use_dynamic_shapes"])
        self.assertFalse(config["trt_collect_shape_range_info"])
        self.assertTrue(config["trt_discard_cached_shape_range_info"])
        self.assertFalse(config["trt_allow_rebuild_at_runtime"])
        self.assertEqual(config["mkldnn_cache_capacity"], 24)
        self.assertEqual(
            option._cfg,
            {
                "enable_new_ir": False,
                "enable_cinn": True,
                "trt_use_dynamic_shapes": False,
                "trt_collect_shape_range_info": False,
                "trt_discard_cached_shape_range_info": True,
                "trt_allow_rebuild_at_runtime": False,
                "mkldnn_cache_capacity": 24,
            },
        )

        option = module._LocalPaddlePredictorOption(
            run_mode="mkldnn",
            cpu_threads=4,
            enable_new_ir=False,
        )

        option.setdefault_by_model_name("ocr-model")

        self.assertEqual(option._cfg["model_name"], "ocr-model")
        self.assertEqual(option._cfg["run_mode"], "mkldnn")
        self.assertEqual(option._cfg["device_type"], "cpu")
        self.assertIsNone(option._cfg["device_id"])
        self.assertEqual(option._cfg["cpu_threads"], 4)
        self.assertEqual(option._cfg["delete_pass"], [])
        self.assertFalse(option._cfg["enable_new_ir"])
        self.assertFalse(option._cfg["enable_cinn"])
        self.assertEqual(option._cfg["mkldnn_cache_capacity"], 10)

    def test_local_paddle_predictor_option_setdefault_by_model_name_preserves_explicit_zero_mkldnn_cache_capacity(self):
        option = module._LocalPaddlePredictorOption(mkldnn_cache_capacity=0)

        option.setdefault_by_model_name("ocr-model")

        self.assertEqual(option._cfg["model_name"], "ocr-model")
        self.assertEqual(option._cfg["run_mode"], "paddle")
        self.assertEqual(option._cfg["device_type"], "cpu")
        self.assertIsNone(option._cfg["device_id"])
        self.assertEqual(option._cfg["mkldnn_cache_capacity"], 0)
        self.assertEqual(option.mkldnn_cache_capacity, 0)

    def test_local_paddle_predictor_option_setdefault_by_model_name_preserves_explicit_delete_pass(self):
        option = module._LocalPaddlePredictorOption(delete_pass=["skip_a", "skip_b"])

        option.setdefault_by_model_name("ocr-model")

        self.assertEqual(option._cfg["model_name"], "ocr-model")
        self.assertEqual(option._cfg["run_mode"], "paddle")
        self.assertEqual(option._cfg["device_type"], "cpu")
        self.assertIsNone(option._cfg["device_id"])
        self.assertEqual(option._cfg["delete_pass"], ["skip_a", "skip_b"])
        self.assertEqual(option.delete_pass, ["skip_a", "skip_b"])

    def test_local_paddle_predictor_option_setdefault_by_model_name_preserves_existing_model_name(self):
        option = module._LocalPaddlePredictorOption(run_mode="mkldnn")
        option._update("model_name", "cached-model")

        option.setdefault_by_model_name("ocr-model")

        self.assertEqual(option._cfg["model_name"], "cached-model")
        self.assertEqual(option._cfg["run_mode"], "mkldnn")
        self.assertEqual(option._cfg["device_type"], "cpu")
        self.assertIsNone(option._cfg["device_id"])
        self.assertEqual(option._cfg["cpu_threads"], 10)
        self.assertEqual(option._cfg["delete_pass"], [])

    def test_local_paddle_predictor_option_setdefault_by_model_name_preserves_explicit_cpu_device_id(self):
        option = module._LocalPaddlePredictorOption(device_type="cpu", device_id=7)

        option.setdefault_by_model_name("ocr-model")

        self.assertEqual(option._cfg["model_name"], "ocr-model")
        self.assertEqual(option._cfg["run_mode"], "paddle")
        self.assertEqual(option._cfg["device_type"], "cpu")
        self.assertEqual(option._cfg["device_id"], 7)
        self.assertEqual(option.device_type, "cpu")
        self.assertEqual(option.device_id, 7)

    def test_local_paddle_predictor_option_setdefault_by_model_name_preserves_shape_info_filename_alias_value(self):
        option = module._LocalPaddlePredictorOption(shape_info_filename="cached.pbtxt")

        option.setdefault_by_model_name("ocr-model")

        self.assertEqual(option._cfg["model_name"], "ocr-model")
        self.assertEqual(option._cfg["trt_shape_range_info_path"], "cached.pbtxt")
        self.assertEqual(option.shape_info_filename, "cached.pbtxt")
        self.assertNotIn("shape_info_filename", option._cfg)

    def test_parse_device_parses_and_normalizes_device_strings(self):
        self.assertEqual(module.parse_device(" GPU:3,7 "), ("gpu", [3, 7]))
        self.assertEqual(module.parse_device("cpu"), ("cpu", None))

    def test_parse_device_rejects_cpu_device_ids(self):
        with self.assertRaisesRegex(ValueError, "No Device ID should be specified for CPUs"):
            module.parse_device("cpu:0")

    def test_get_default_device_returns_cpu_when_paddle_cuda_check_is_unavailable(self):
        with patch.object(module, "_load_module_without_package_init", side_effect=ImportError):
            self.assertEqual(module.get_default_device(), "cpu")

    def test_get_default_device_returns_gpu_zero_when_cuda_is_available(self):
        fake_paddle_core = module.types.SimpleNamespace(is_compiled_with_cuda=lambda: True)

        with patch.object(module, "_load_module_without_package_init", return_value=fake_paddle_core):
            self.assertEqual(module.get_default_device(), "gpu:0")

    def test_set_env_for_device_type_updates_cuda_visible_devices(self):
        with patch.dict(module.os.environ, {"CUDA_VISIBLE_DEVICES": "0,1"}, clear=False):
            module.set_env_for_device_type("cpu")
            self.assertEqual(module.os.environ["CUDA_VISIBLE_DEVICES"], "")
            module.set_env_for_device_type("gpu")
            self.assertNotIn("CUDA_VISIBLE_DEVICES", module.os.environ)

    def test_get_default_run_mode_returns_paddle_for_local_shim(self):
        self.assertEqual(module.get_default_run_mode("ocr-model", "cpu"), "paddle")
        self.assertEqual(module.get_default_run_mode("ocr-model", "gpu"), "paddle")
        self.assertEqual(module.get_default_run_mode(None, "cpu"), "paddle")

    def test_local_paddle_predictor_option_device_type_setter_updates_runtime_env(self):
        option = module._LocalPaddlePredictorOption()

        with patch.dict(module.os.environ, {}, clear=True):
            option.device_type = "gpu"
            self.assertEqual(module.os.environ["FLAGS_enable_pir_api"], "1")
            option.device_type = "cpu"
            self.assertEqual(module.os.environ["FLAGS_enable_pir_api"], "1")
            self.assertEqual(module.os.environ["CUDA_VISIBLE_DEVICES"], "")

    def test_loader_helpers_ensure_paddlex_shim_before_loading_runtime_modules(self):
        cases = [
            ("_load_paddleocr_class", "paddleocr._pipelines.ocr", "PaddleOCR"),
            ("_load_pdf_reader_class", "paddlex.inference.utils.io.readers", "PDFReader"),
        ]

        for helper_name, module_name, attribute_name in cases:
            with self.subTest(helper_name=helper_name):
                expected_class = object()
                fake_loaded_module = module.types.SimpleNamespace(**{attribute_name: expected_class})
                call_order = []

                with patch.object(module, "_ensure_paddlex_shim", side_effect=lambda: call_order.append("ensure")), \
                     patch.object(
                         module,
                         "_load_module_without_package_init",
                         side_effect=lambda requested_module_name, fake_loaded_module=fake_loaded_module: call_order.append(requested_module_name) or fake_loaded_module,
                     ):
                    result = getattr(module, helper_name)()

                self.assertIs(result, expected_class)
                self.assertEqual(call_order, ["ensure", module_name])

    def test_get_bundled_model_dir_returns_model_path_when_present(self):
        resource_root = self.root / "resources"
        model_dir = resource_root / "models" / module.TEXT_DETECTION_MODEL_NAME
        model_dir.mkdir(parents=True, exist_ok=True)
        (model_dir / "inference.yml").write_text("ok", encoding="utf-8")

        with patch.object(module, "get_runtime_resource_dir", return_value=resource_root):
            self.assertEqual(module.get_bundled_model_dir(module.TEXT_DETECTION_MODEL_NAME), model_dir)

    def test_get_bundled_model_dir_requires_populated_model_directory(self):
        resource_root = self.root / "resources"
        model_name = module.TEXT_DETECTION_MODEL_NAME
        empty_model_dir = resource_root / "models" / model_name
        empty_model_dir.mkdir(parents=True, exist_ok=True)

        with patch.object(module, "get_runtime_resource_dir", return_value=resource_root):
            self.assertIsNone(module.get_bundled_model_dir(model_name))

            (empty_model_dir / "inference.yml").write_text("ok", encoding="utf-8")

            self.assertEqual(module.get_bundled_model_dir(model_name), empty_model_dir)

    def test_get_model_cache_dir_uses_paddlex_official_models_under_home(self):
        fake_home = self.root / "fake-home"

        with patch.object(module.Path, "home", return_value=fake_home):
            self.assertEqual(
                module.get_model_cache_dir(),
                fake_home / ".paddlex" / "official_models",
            )

    def test_get_cached_model_dir_requires_populated_model_directory(self):
        cache_root = self.root / "cache"
        model_name = module.TEXT_DETECTION_MODEL_NAME
        empty_model_dir = cache_root / model_name
        empty_model_dir.mkdir(parents=True, exist_ok=True)

        with patch.object(module, "get_model_cache_dir", return_value=cache_root):
            self.assertIsNone(module.get_cached_model_dir(model_name))

            (empty_model_dir / "inference.yml").write_text("ok", encoding="utf-8")

            self.assertEqual(module.get_cached_model_dir(model_name), empty_model_dir)

    def test_get_local_model_dir_prefers_bundled_model_before_cache(self):
        bundled_model_dir = self.root / "bundled-model"
        cached_model_dir = self.root / "cached-model"

        with patch.object(module, "get_bundled_model_dir", return_value=bundled_model_dir), \
             patch.object(module, "get_cached_model_dir", return_value=cached_model_dir) as cached_mock:
            self.assertEqual(module.get_local_model_dir(module.TEXT_DETECTION_MODEL_NAME), bundled_model_dir)

        cached_mock.assert_not_called()

    def test_get_local_model_dir_uses_cache_only_for_source_runtime(self):
        cached_model_dir = self.root / "cached-model"

        with patch.object(module, "get_bundled_model_dir", return_value=None), \
             patch.object(module, "is_packaged_runtime", return_value=False), \
             patch.object(module, "get_cached_model_dir", return_value=cached_model_dir) as cached_mock:
            self.assertEqual(module.get_local_model_dir(module.TEXT_DETECTION_MODEL_NAME), cached_model_dir)

        cached_mock.assert_called_once_with(module.TEXT_DETECTION_MODEL_NAME)

        with patch.object(module, "get_bundled_model_dir", return_value=None), \
             patch.object(module, "is_packaged_runtime", return_value=True), \
             patch.object(module, "get_cached_model_dir") as cached_mock:
            self.assertIsNone(module.get_local_model_dir(module.TEXT_DETECTION_MODEL_NAME))

        cached_mock.assert_not_called()

    def test_get_local_model_dir_falls_back_to_populated_cache_when_source_bundle_is_empty(self):
        resource_root = self.root / "resources"
        cache_root = self.root / "cache"
        model_name = module.TEXT_DETECTION_MODEL_NAME
        bundled_model_dir = resource_root / "models" / model_name
        cached_model_dir = cache_root / model_name
        bundled_model_dir.mkdir(parents=True, exist_ok=True)
        cached_model_dir.mkdir(parents=True, exist_ok=True)
        (cached_model_dir / "inference.yml").write_text("ok", encoding="utf-8")

        with patch.object(module, "get_runtime_resource_dir", return_value=resource_root), \
             patch.object(module, "get_model_cache_dir", return_value=cache_root), \
             patch.object(module, "is_packaged_runtime", return_value=False):
            self.assertEqual(module.get_local_model_dir(model_name), cached_model_dir)

    def test_get_local_model_dir_ignores_cache_for_packaged_runtime_when_bundle_is_empty(self):
        resource_root = self.root / "resources"
        cache_root = self.root / "cache"
        model_name = module.TEXT_DETECTION_MODEL_NAME
        bundled_model_dir = resource_root / "models" / model_name
        cached_model_dir = cache_root / model_name
        bundled_model_dir.mkdir(parents=True, exist_ok=True)
        cached_model_dir.mkdir(parents=True, exist_ok=True)
        (cached_model_dir / "inference.yml").write_text("ok", encoding="utf-8")

        with patch.object(module, "get_runtime_resource_dir", return_value=resource_root), \
             patch.object(module, "get_model_cache_dir", return_value=cache_root), \
             patch.object(module, "is_packaged_runtime", return_value=True):
            self.assertIsNone(module.get_local_model_dir(model_name))

    def test_is_populated_directory_distinguishes_empty_populated_and_unreadable_dirs(self):
        empty_dir = self.root / "empty-dir"
        empty_dir.mkdir(parents=True, exist_ok=True)
        populated_dir = self.root / "populated-dir"
        populated_dir.mkdir(parents=True, exist_ok=True)
        (populated_dir / "marker.txt").write_text("ok", encoding="utf-8")
        unreadable_dir = Mock(spec=Path)
        unreadable_dir.is_dir.return_value = True
        unreadable_dir.iterdir.side_effect = OSError("boom")

        self.assertFalse(module._is_populated_directory(self.root / "missing-dir"))
        self.assertFalse(module._is_populated_directory(empty_dir))
        self.assertTrue(module._is_populated_directory(populated_dir))
        self.assertFalse(module._is_populated_directory(unreadable_dir))

    def test_get_recognition_model_name_normalizes_language_input(self):
        self.assertEqual(
            module.get_recognition_model_name(" EN "),
            module.LANGUAGE_RECOGNITION_MODEL_NAMES["en"],
        )
        self.assertEqual(
            module.get_recognition_model_name("JaPaN"),
            module.LANGUAGE_RECOGNITION_MODEL_NAMES["japan"],
        )
        self.assertIsNone(module.get_recognition_model_name("unknown"))
        self.assertIsNone(module.get_recognition_model_name("   "))
        self.assertIsNone(module.get_recognition_model_name(None))

    def test_get_missing_bundled_model_names_reports_absent_models(self):
        rec_model_dir = self.root / "resources" / "models" / module.TEXT_RECOGNITION_MODEL_NAME

        with patch.object(module, "get_bundled_model_dir", side_effect=[None, rec_model_dir]):
            self.assertEqual(module.get_missing_bundled_model_names(), [module.TEXT_DETECTION_MODEL_NAME])

    def test_get_missing_bundled_model_names_reports_default_chinese_models_in_order(self):
        with patch.object(module, "get_bundled_model_dir", side_effect=[None, None]):
            self.assertEqual(
                module.get_missing_bundled_model_names(),
                [
                    module.TEXT_DETECTION_MODEL_NAME,
                    module.TEXT_RECOGNITION_MODEL_NAME,
                ],
            )

    def test_get_missing_bundled_model_names_normalizes_japan_before_reporting_missing_models(self):
        with patch.object(module, "get_bundled_model_dir", side_effect=[None, None]):
            self.assertEqual(
                module.get_missing_bundled_model_names(language=" JaPaN "),
                [
                    module.TEXT_DETECTION_MODEL_NAME,
                    module.TEXT_RECOGNITION_MODEL_NAME,
                ],
            )

    def test_get_missing_bundled_model_names_skips_unknown_language_when_detection_is_bundled(self):
        det_model_dir = self.root / "resources" / "models" / module.TEXT_DETECTION_MODEL_NAME

        with patch.object(module, "get_bundled_model_dir", return_value=det_model_dir):
            self.assertEqual(module.get_missing_bundled_model_names(language="unknown"), [])

    def test_get_missing_bundled_model_names_treats_none_like_unknown_language(self):
        with patch.object(module, "get_bundled_model_dir", return_value=None):
            self.assertEqual(
                module.get_missing_bundled_model_names(language=None),
                [module.TEXT_DETECTION_MODEL_NAME],
            )
            self.assertEqual(
                module.get_missing_bundled_model_names(language="   "),
                [module.TEXT_DETECTION_MODEL_NAME],
            )

    def test_get_missing_bundled_model_names_reports_english_models_in_order(self):
        with patch.object(module, "get_bundled_model_dir", side_effect=[None, None]):
            self.assertEqual(
                module.get_missing_bundled_model_names(language="en"),
                [
                    module.TEXT_DETECTION_MODEL_NAME,
                    module.get_recognition_model_name("en"),
                ],
            )

    def test_get_missing_bundled_model_names_normalizes_language_before_reporting_missing_models(self):
        with patch.object(module, "get_bundled_model_dir", side_effect=[None, None]):
            self.assertEqual(
                module.get_missing_bundled_model_names(language=" EN "),
                [
                    module.TEXT_DETECTION_MODEL_NAME,
                    module.get_recognition_model_name("en"),
                ],
            )

    def test_get_required_model_names_matches_supported_language_variants(self):
        self.assertEqual(
            module.get_required_model_names(language="ch"),
            [module.TEXT_DETECTION_MODEL_NAME, module.TEXT_RECOGNITION_MODEL_NAME],
        )
        self.assertEqual(
            module.get_required_model_names(language="japan"),
            [module.TEXT_DETECTION_MODEL_NAME, module.TEXT_RECOGNITION_MODEL_NAME],
        )
        self.assertEqual(
            module.get_required_model_names(language=" JaPaN "),
            [module.TEXT_DETECTION_MODEL_NAME, module.TEXT_RECOGNITION_MODEL_NAME],
        )
        self.assertEqual(
            module.get_required_model_names(language="en"),
            [module.TEXT_DETECTION_MODEL_NAME, module.get_recognition_model_name("en")],
        )
        self.assertEqual(
            module.get_required_model_names(language=" EN "),
            [module.TEXT_DETECTION_MODEL_NAME, module.get_recognition_model_name("en")],
        )
        self.assertEqual(
            module.get_required_model_names(language="korean"),
            [module.TEXT_DETECTION_MODEL_NAME, module.get_recognition_model_name("korean")],
        )
        self.assertEqual(
            module.get_required_model_names(language="latin"),
            [module.TEXT_DETECTION_MODEL_NAME, module.get_recognition_model_name("latin")],
        )

    def test_get_required_model_names_skips_unknown_language(self):
        self.assertEqual(
            module.get_required_model_names(language="unknown"),
            [module.TEXT_DETECTION_MODEL_NAME],
        )
        self.assertEqual(
            module.get_required_model_names(language=None),
            [module.TEXT_DETECTION_MODEL_NAME],
        )
        self.assertEqual(
            module.get_required_model_names(language="   "),
            [module.TEXT_DETECTION_MODEL_NAME],
        )

    def test_build_bat_includes_packaged_model_directories_for_offline_runtime(self):
        build_script = (Path(module.__file__).resolve().parent / "build.bat").read_text(encoding="utf-8")
        # build.bat uses %DET_MODEL_NAME% (set dynamically per build profile), so check the variable reference
        self.assertIn("models/%DET_MODEL_NAME%", build_script)
        # Recognition model is always mobile_rec
        self.assertIn(f"models/{module.TEXT_RECOGNITION_MODEL_NAME}", build_script)

        unexpected_model_names = {
            module.get_recognition_model_name("en"),
            module.get_recognition_model_name("korean"),
            module.get_recognition_model_name("latin"),
        }

        for model_name in unexpected_model_names:
            with self.subTest(model_name=model_name):
                self.assertNotIn(f"models/{model_name}", build_script)

    def test_build_bat_includes_key_runtime_resources_and_nuitka_options(self):
        build_script = (Path(module.__file__).resolve().parent / "build.bat").read_text(encoding="utf-8")
        expected_snippets = [
            'set "OUTPUT_DIR=release"',
            'if not defined APP_NAME set "APP_NAME=OCRFinder"',
            "--standalone",
            "--enable-plugin=tk-inter",
            "--windows-console-mode=disable",
            '--output-dir="%OUTPUT_DIR%"',
            '--output-filename="%APP_NAME%.exe"',
            "--include-module=paddlex.utils.config",
            "--include-module=paddlex.inference.models.common.static_infer",
            "--include-module=paddlex.inference.utils.pp_option",
            "--include-module=paddle.base.core",
        ]
        unexpected_snippets = [
            "README.md=README.md",
            "requirements.txt=requirements.txt",
            "--include-package=rapidocr_onnxruntime",
            "--include-module=onnxruntime",
            "rapidocr_onnxruntime/config.yaml",
            "rapidocr_onnxruntime/models",
            "--include-package=onnxruntime",
            "--include-package=tokenizers",
            "--include-module=tokenizers",
        ]

        for snippet in expected_snippets:
            with self.subTest(snippet=snippet):
                self.assertIn(snippet, build_script)

        for snippet in unexpected_snippets:
            with self.subTest(snippet=snippet):
                self.assertNotIn(snippet, build_script)

    def test_build_bat_checks_cached_models_needed_for_offline_packaging(self):
        build_script = (Path(module.__file__).resolve().parent / "build.bat").read_text(encoding="utf-8")
        expected_cache_variables = [
            "DET_MODEL_DIR",
            "REC_CH_MODEL_DIR",
        ]
        unexpected_cache_variables = [
            "FORMULA_MODEL_DIR",
            "REC_EN_MODEL_DIR",
            "REC_KOREAN_MODEL_DIR",
            "REC_LATIN_MODEL_DIR",
        ]

        for cache_variable in expected_cache_variables:
            with self.subTest(cache_variable=cache_variable):
                self.assertIn(f'if not exist "%{cache_variable}%"', build_script)

        for cache_variable in unexpected_cache_variables:
            with self.subTest(cache_variable=cache_variable):
                self.assertNotIn(f'if not exist "%{cache_variable}%"', build_script)

    def test_build_bat_documents_japanese_model_reuse(self):
        build_script = (Path(module.__file__).resolve().parent / "build.bat").read_text(encoding="utf-8")

        self.assertIn("Japanese PaddleOCR reuses PP-OCRv5_mobile_rec", build_script)
        self.assertNotIn("REC_JAPAN_MODEL_DIR", build_script)

    def test_build_bat_reports_runtime_variant_that_will_be_bundled(self):
        build_script = (Path(module.__file__).resolve().parent / "build.bat").read_text(encoding="utf-8")
        expected_snippets = [
            "Detecting installed Paddle runtime package",
            "import paddle.base.core as core; print('[INFO] Paddle CUDA build:'",
            "Paddle CUDA build:",
            "Could not inspect installed Paddle runtime package",
            "currently installed local Paddle runtime variant from venv",
            "OCR_RUNTIME_TARGET=gpu",
            "bundle that local Paddle GPU runtime as well",
        ]

        for snippet in expected_snippets:
            with self.subTest(snippet=snippet):
                self.assertIn(snippet, build_script)

    def test_build_bat_reports_package_dir_summary_and_optional_license_copy(self):
        build_script = (Path(module.__file__).resolve().parent / "build.bat").read_text(encoding="utf-8")
        expected_snippets = [
            'set "PACKAGE_DIR="',
            "glob('*.dist')",
            'set "PACKAGE_DIR=%%D"',
            'if not defined PACKAGE_DIR (',
            "Build completed but no Nuitka .dist folder was found under %OUTPUT_DIR%.",
            'for /d %%D in ("%OUTPUT_DIR%\\*.build") do if exist "%%~fD" rmdir /s /q "%%~fD"',
            'if exist LICENSE copy LICENSE "%PACKAGE_DIR%\\" >nul 2>&1',
            "echo Build Summary",
            "echo Package folder: %PACKAGE_DIR%",
            "echo Executable: %PACKAGE_DIR%\\%APP_NAME%.exe",
            'if exist "%PACKAGE_DIR%\\%APP_NAME%.exe" (',
            "echo Executable size: %%~zF bytes",
        ]

        for snippet in expected_snippets:
            with self.subTest(snippet=snippet):
                self.assertIn(snippet, build_script)

    def test_build_cpu_bat_enforces_cpu_runtime_and_size_limit(self):
        build_script = (Path(module.__file__).resolve().parent / "build_cpu.bat").read_text(encoding="utf-8")
        expected_snippets = [
            'set "OUTPUT_DIR=release"',
            'set "APP_NAME=OCRFinder"',
            'set "PACKAGE_SIZE_BYTES="',
            'set "MAX_PACKAGE_BYTES=209715200"',
            "Please run setup_cpu_venv.bat first to create the environment.",
            "Verifying CPU Paddle runtime before packaging",
            "import paddle.base.core as core; print('[INFO] Paddle CUDA build:'",
            'python -c "import paddle.base.core as core, sys; sys.exit(0 if core.is_compiled_with_cuda() else 1)" >nul 2>&1',
            "build_cpu.bat requires the baseline CPU Paddle runtime.",
            "Re-run setup_cpu_venv.bat or recreate venv before packaging.",
            'call "%~dp0build.bat"',
            "glob('*.dist')",
            "package_dir.rglob('*')",
            "CPU package exceeds the 200MB limit enforced by build_cpu.bat.",
            "CPU package size is within the 200MB limit.",
        ]

        for snippet in expected_snippets:
            with self.subTest(snippet=snippet):
                self.assertIn(snippet, build_script)

    def test_build_gpu_bat_requires_local_gpu_runtime_and_delegates_to_shared_build(self):
        build_script = (Path(module.__file__).resolve().parent / "build_gpu.bat").read_text(encoding="utf-8")
        expected_snippets = [
            "Please run setup_gpu_venv.bat first to create the environment.",
            "Verifying GPU Paddle runtime before packaging",
            "import paddle.base.core as core; print('[INFO] Paddle CUDA build:'",
            'python -c "import paddle.base.core as core, sys; sys.exit(0 if core.is_compiled_with_cuda() else 1)" >nul 2>&1',
            "build_gpu.bat requires a locally installed Paddle GPU runtime.",
            "Run setup_gpu_venv.bat with OCR_PADDLE_GPU_WHEEL set to a local GPU wheel first.",
            'set "APP_NAME=OCRFinder_GPU"',
            'call "%~dp0build.bat"',
            "GPU package build completed with the local Paddle GPU runtime.",
        ]

        for snippet in expected_snippets:
            with self.subTest(snippet=snippet):
                self.assertIn(snippet, build_script)

    def test_batch_scripts_skip_pause_during_non_interactive_runs(self):
        script_texts = {
            "build.bat": (Path(module.__file__).resolve().parent / "build.bat").read_text(encoding="utf-8"),
            "build_cpu.bat": (Path(module.__file__).resolve().parent / "build_cpu.bat").read_text(encoding="utf-8"),
            "build_gpu.bat": (Path(module.__file__).resolve().parent / "build_gpu.bat").read_text(encoding="utf-8"),
            "setup_venv.bat": (Path(module.__file__).resolve().parent / "setup_venv.bat").read_text(encoding="utf-8"),
            "setup_cpu_venv.bat": (Path(module.__file__).resolve().parent / "setup_cpu_venv.bat").read_text(encoding="utf-8"),
            "setup_gpu_venv.bat": (Path(module.__file__).resolve().parent / "setup_gpu_venv.bat").read_text(encoding="utf-8"),
        }
        expected_snippets = [
            'setlocal EnableExtensions',
            'set "OCR_FINDER_SKIP_PAUSE="',
            'if defined OCR_FINDER_NO_PAUSE set "OCR_FINDER_SKIP_PAUSE=1"',
            'if defined CI set "OCR_FINDER_SKIP_PAUSE=1"',
            '[Console]::IsInputRedirected -or [Console]::IsOutputRedirected -or [Console]::IsErrorRedirected',
            'call :maybe_pause',
            ':maybe_pause',
            'if defined OCR_FINDER_SKIP_PAUSE goto :eof',
        ]

        for script_name, script_text in script_texts.items():
            with self.subTest(script_name=script_name):
                for snippet in expected_snippets:
                    self.assertIn(snippet, script_text)
                self.assertEqual([line.strip() for line in script_text.splitlines()].count("pause"), 1)

    def test_setup_venv_defaults_to_cpu_runtime_target_and_validates_supported_values(self):
        setup_script = (Path(module.__file__).resolve().parent / "setup_venv.bat").read_text(encoding="utf-8")
        expected_snippets = [
            'if not defined OCR_RUNTIME_TARGET set "OCR_RUNTIME_TARGET=cpu"',
            'if /I not "%OCR_RUNTIME_TARGET%"=="cpu" if /I not "%OCR_RUNTIME_TARGET%"=="gpu" (',
            'echo [ERROR] Unsupported OCR_RUNTIME_TARGET: %OCR_RUNTIME_TARGET%',
        ]

        for snippet in expected_snippets:
            with self.subTest(snippet=snippet):
                self.assertIn(snippet, setup_script)

    def test_setup_venv_requires_local_gpu_wheels_for_gpu_target(self):
        setup_script = (Path(module.__file__).resolve().parent / "setup_venv.bat").read_text(encoding="utf-8")
        expected_snippets = [
            "OCR_PADDLE_GPU_WHEEL is required when OCR_RUNTIME_TARGET=gpu",
            "Point it to a local Paddle GPU wheel before running setup_venv.bat.",
            'if not exist "%OCR_PADDLE_GPU_WHEEL%" (',
        ]

        for snippet in expected_snippets:
            with self.subTest(snippet=snippet):
                self.assertIn(snippet, setup_script)

    def test_setup_venv_installs_local_gpu_runtimes_and_verifies_cuda_support(self):
        setup_script = (Path(module.__file__).resolve().parent / "setup_venv.bat").read_text(encoding="utf-8")
        expected_snippets = [
            "Replacing CPU Paddle runtime with local GPU wheel",
            'pip uninstall -y paddlepaddle paddlepaddle-gpu paddlepaddle_gpu >nul 2>&1',
            'pip install --no-deps "%OCR_PADDLE_GPU_WHEEL%"',
            "import paddle.base.core as core; print('[INFO] Paddle CUDA build:'",
            "core.is_compiled_with_cuda()",
            "The offline EXE build will bundle that local Paddle GPU runtime when available.",
        ]

        for snippet in expected_snippets:
            with self.subTest(snippet=snippet):
                self.assertIn(snippet, setup_script)

    def test_setup_cpu_venv_resets_and_verifies_baseline_cpu_runtime(self):
        setup_script = (Path(module.__file__).resolve().parent / "setup_cpu_venv.bat").read_text(encoding="utf-8")
        expected_snippets = [
            "Resetting Paddle runtime packages to the baseline CPU variant",
            'pip uninstall -y paddlepaddle paddlepaddle-gpu paddlepaddle_gpu >nul 2>&1',
            "Verifying CPU Paddle runtime",
            "import paddle.base.core as core; print('[INFO] Paddle CUDA build:'",
            'python -c "import paddle.base.core as core, sys; sys.exit(0 if core.is_compiled_with_cuda() else 1)" >nul 2>&1',
            "CPU setup expects the baseline CPU Paddle runtime from requirements.txt.",
            "run: build_cpu.bat",
        ]
        unexpected_snippets = [
            "OCR_PADDLE_GPU_WHEEL",
            "OCR_RUNTIME_TARGET=gpu",
        ]

        for snippet in expected_snippets:
            with self.subTest(snippet=snippet):
                self.assertIn(snippet, setup_script)

        for snippet in unexpected_snippets:
            with self.subTest(snippet=snippet):
                self.assertNotIn(snippet, setup_script)

    def test_setup_gpu_venv_requires_local_gpu_wheel_and_delegates_to_legacy_setup(self):
        setup_script = (Path(module.__file__).resolve().parent / "setup_gpu_venv.bat").read_text(encoding="utf-8")
        expected_snippets = [
            "OCR_PADDLE_GPU_WHEEL is required for setup_gpu_venv.bat",
            "Point it to a local Paddle GPU wheel before running setup_gpu_venv.bat.",
            'if not exist "%OCR_PADDLE_GPU_WHEEL%" (',
            'set "OCR_RUNTIME_TARGET=gpu"',
            'call setup_venv.bat',
            'set "EXIT_CODE=%ERRORLEVEL%"',
            'endlocal & exit /b %EXIT_CODE%',
        ]

        for snippet in expected_snippets:
            with self.subTest(snippet=snippet):
                self.assertIn(snippet, setup_script)

    def test_setup_venv_warms_offline_model_cache_for_supported_languages(self):
        setup_script = (Path(module.__file__).resolve().parent / "setup_venv.bat").read_text(encoding="utf-8")

        self.assertIn("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", setup_script)

        expected_languages = {
            "ch": module.TEXT_RECOGNITION_MODEL_NAME,
            "japan": module.TEXT_RECOGNITION_MODEL_NAME,
        }

        for language, recognition_model in expected_languages.items():
            with self.subTest(language=language):
                self.assertIn(f"Caching PaddleOCR models for {language}...", setup_script)
                self.assertIn(
                    f"PaddleOCR(lang='{language}', text_detection_model_name='{module.TEXT_DETECTION_MODEL_NAME}', "
                    f"text_recognition_model_name='{recognition_model}', use_doc_orientation_classify=False, "
                    "use_doc_unwarping=False, use_textline_orientation=False)",
                    setup_script,
                )

    def test_setup_venv_keeps_paddle_cache_warmup_pinned_to_offline_safe_flags(self):
        setup_script = (Path(module.__file__).resolve().parent / "setup_venv.bat").read_text(encoding="utf-8")
        paddle_warmups = [
            line
            for line in setup_script.splitlines()
            if "from paddleocr import PaddleOCR" in line and "Caching PaddleOCR models for" in line
        ]
        expected_languages = {"ch", "japan"}

        self.assertEqual(len(expected_languages), len(paddle_warmups))

        for language in expected_languages:
            self.assertTrue(
                any(f"Caching PaddleOCR models for {language}..." in warmup_line for warmup_line in paddle_warmups)
            )

        for warmup_line in paddle_warmups:
            with self.subTest(warmup_line=warmup_line):
                self.assertIn("use_doc_orientation_classify=False", warmup_line)
                self.assertIn("use_doc_unwarping=False", warmup_line)
                self.assertIn("use_textline_orientation=False", warmup_line)

    def test_initialize_backends_bootstraps_shim_before_paddle_init(self):
        finder = self.make_finder(build_profile="gpu", backend="paddle")
        call_order = []

        def fake_bootstrap():
            call_order.append("bootstrap")

        def fake_initialize_paddle_backend():
            call_order.append("paddle")
            return True

        with patch.object(module, "_bootstrap_local_paddlex_pp_option_shim", side_effect=fake_bootstrap) as bootstrap_mock, \
             patch.object(module.OCRFinder, "_initialize_paddle_backend", side_effect=fake_initialize_paddle_backend) as initialize_mock:
            result = finder.initialize_backends()

        self.assertTrue(result)
        self.assertEqual(call_order, ["bootstrap", "paddle"])
        bootstrap_mock.assert_called_once_with()
        initialize_mock.assert_called_once_with()

    def test_initialize_backends_dispatches_selected_rapidocr_backend(self):
        finder = self.make_finder(build_profile="gpu", backend="rapidocr")

        with patch.object(module, "_bootstrap_local_paddlex_pp_option_shim") as bootstrap_mock, \
             patch.object(module.OCRFinder, "_initialize_paddle_backend") as paddle_mock, \
             patch.object(module.OCRFinder, "_initialize_rapidocr_backend", return_value=True) as rapidocr_mock:
            result = finder.initialize_backends()

        self.assertTrue(result)
        bootstrap_mock.assert_not_called()
        paddle_mock.assert_not_called()
        rapidocr_mock.assert_called_once_with()

    def test_initialize_backends_rejects_backend_not_allowed_by_build_profile(self):
        finder = self.make_finder(build_profile="cpu", backend="rapidocr")

        with patch.object(finder, "error") as error_mock, \
             patch.object(module, "_bootstrap_local_paddlex_pp_option_shim") as bootstrap_mock, \
             patch.object(module.OCRFinder, "_initialize_rapidocr_backend") as rapidocr_mock:
            result = finder.initialize_backends()

        self.assertFalse(result)
        bootstrap_mock.assert_not_called()
        rapidocr_mock.assert_not_called()
        error_mock.assert_called_once_with(
            "Build profile 'cpu' does not support OCR backend 'rapidocr'. Allowed backends: paddle."
        )

    def test_initialize_backends_reports_packaged_dependency_issue_without_install_hint(self):
        finder = self.make_finder()

        with patch.object(module, "Image", None), \
             patch.object(module, "is_packaged_runtime", return_value=True), \
             patch.object(finder, "error") as error_mock:
            result = finder.initialize_backends()

        self.assertFalse(result)
        self.assertEqual(error_mock.call_count, 1)
        self.assertIn("bundled Python dependencies are missing", error_mock.call_args[0][0])
        self.assertNotIn("pip install", error_mock.call_args[0][0])

    def test_initialize_backends_requires_bundled_models_in_packaged_runtime(self):
        finder = self.make_finder()

        with patch.object(module, "Image", object()), \
             patch.object(module, "is_packaged_runtime", return_value=True), \
             patch.object(module, "get_bundled_model_dir", return_value=None), \
             patch.object(finder, "error") as error_mock:
            result = finder.initialize_backends()

        self.assertFalse(result)
        self.assertEqual(error_mock.call_count, 1)
        self.assertIn("Packaged OCR models are missing", error_mock.call_args[0][0])
        self.assertIn("end users should not install Python or download models", error_mock.call_args[0][0])

        finder = self.make_finder()
        fake_ocr_instance = Mock()
        fake_ocr_class = Mock(return_value=fake_ocr_instance)
        det_model_dir = self.root / "cache" / module.TEXT_DETECTION_MODEL_NAME
        rec_model_dir = self.root / "cache" / module.TEXT_RECOGNITION_MODEL_NAME

        with patch.object(module, "Image", object()), \
             patch.object(module, "PaddleOCR", fake_ocr_class), \
             patch.object(module, "get_local_model_dir", side_effect=[det_model_dir, rec_model_dir]), \
             patch.object(finder, "_can_use_paddle_gpu", return_value=False):
            result = finder.initialize_backends()

        self.assertTrue(result)
        self.assertIs(finder.ocr, fake_ocr_instance)
        self.assertEqual(finder.resolved_device, "cpu")
        fake_ocr_class.assert_called_once_with(
            lang="ch",
            text_detection_model_name=module.TEXT_DETECTION_MODEL_NAME,
            text_detection_model_dir=str(det_model_dir),
            text_recognition_model_name=module.TEXT_RECOGNITION_MODEL_NAME,
            text_recognition_model_dir=str(rec_model_dir),
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
            device="cpu",
        )

    def test_initialize_backends_uses_bundled_models_when_available(self):
        finder = self.make_finder(verbose=True)
        fake_ocr_instance = Mock()
        fake_ocr_class = Mock(return_value=fake_ocr_instance)
        det_model_dir = self.root / "models" / module.TEXT_DETECTION_MODEL_NAME
        rec_model_dir = self.root / "models" / module.TEXT_RECOGNITION_MODEL_NAME

        with patch.object(module, "Image", object()), \
             patch.object(module, "PaddleOCR", fake_ocr_class), \
             patch.object(module, "get_bundled_model_dir", side_effect=[det_model_dir, rec_model_dir]), \
             patch.object(finder, "_can_use_paddle_gpu", return_value=False):
            result = finder.initialize_backends()

        self.assertTrue(result)
        self.assertIs(finder.ocr, fake_ocr_instance)
        self.assertEqual(finder.resolved_device, "cpu")
        fake_ocr_class.assert_called_once_with(
            lang="ch",
            text_detection_model_name=module.TEXT_DETECTION_MODEL_NAME,
            text_detection_model_dir=str(det_model_dir),
            text_recognition_model_name=module.TEXT_RECOGNITION_MODEL_NAME,
            text_recognition_model_dir=str(rec_model_dir),
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
            device="cpu",
        )

    def test_run_copies_only_matching_files_without_real_ocr(self):
        source = self.root / "source"
        output = self.root / "output"
        first_image = self.make_file("source/first.png", b"first")
        second_image = self.make_file("source/second.png", b"second")
        finder = self.make_finder(target="你", source=source, output=output)
        stdout = io.StringIO()

        with patch.object(module.OCRFinder, "initialize_backends", return_value=True), \
             patch.object(module.OCRFinder, "get_input_files", return_value=[first_image, second_image]), \
             patch.object(module.OCRFinder, "perform_ocr", side_effect=["这里有你", "完全不匹配"]), \
             redirect_stdout(stdout):
            exit_code = finder.run()

        self.assertEqual(exit_code, 0)
        self.assertEqual([path.name for path in finder.matched_files], ["first.png"])
        self.assertEqual(sorted(path.name for path in output.iterdir()), ["first.png"])
        self.assertEqual(finder.processed_files, 2)
        self.assertEqual(finder.skipped_files, 1)
        self.assertEqual(finder.error_files, 0)
        self.assertIn("Matches found: 1", stdout.getvalue())

    def test_run_emits_events_for_gui_consumers(self):
        source = self.root / "source"
        output = self.root / "output"
        first_image = self.make_file("source/first.png", b"first")
        events = []
        finder = self.make_finder(target="你", source=source, output=output, event_callback=events.append)

        with patch.object(module.OCRFinder, "initialize_backends", return_value=True), \
             patch.object(module.OCRFinder, "get_input_files", return_value=[first_image]), \
             patch.object(module.OCRFinder, "perform_ocr", return_value="这里有你"):
            exit_code = finder.run()

        self.assertEqual(exit_code, 0)
        event_kinds = {event["kind"] for event in events}
        self.assertIn("status", event_kinds)
        self.assertIn("scan_complete", event_kinds)
        self.assertIn("file_start", event_kinds)
        self.assertIn("match", event_kinds)
        self.assertIn("finished", event_kinds)

    def test_run_reports_no_supported_files_without_initializing_backends(self):
        source = self.root / "source"
        output = self.root / "output"
        events = []
        finder = self.make_finder(target="你", source=source, output=output, event_callback=events.append)
        stdout = io.StringIO()

        with patch.object(module.OCRFinder, "get_input_files", return_value=[]), \
             patch.object(module.OCRFinder, "initialize_backends") as initialize_backends_mock, \
             redirect_stdout(stdout):
            exit_code = finder.run()

        self.assertEqual(exit_code, 0)
        initialize_backends_mock.assert_not_called()
        self.assertIn("No supported files found in source directory.", stdout.getvalue())
        self.assertEqual(finder.processed_files, 0)
        self.assertEqual([event["kind"] for event in events], ["status", "status", "finished"])
        self.assertEqual(
            events[-1],
            {
                "kind": "finished",
                "message": "No supported files found.",
                "summary": {
                    "total_files": 0,
                    "processed_files": 0,
                    "matches_found": 0,
                    "skipped_files": 0,
                    "error_files": 0,
                    "elapsed_time": 0.0,
                    "output_dir": str(output.resolve()),
                    "matched_files": [],
                },
            },
        )

    def test_contains_target_text_matches_normalized_multichar_substrings(self):
        finder = self.make_finder(target="你好ABC")
        spaced_finder = self.make_finder(target="  你好   ABC ")

        self.assertTrue(finder.contains_target_text("前缀 你好abc 后缀"))
        self.assertTrue(spaced_finder.contains_target_text("前缀\n你好   ABC\t后缀"))
        self.assertFalse(finder.contains_target_text("今天真好"))
        self.assertFalse(finder.contains_target_text("Version B is ready"))
        self.assertFalse(finder.contains_target_text("世界和平xyz"))

    def test_can_use_paddle_gpu_uses_paddle_base_core_checker(self):
        finder = self.make_finder()
        checker = Mock(return_value=True)
        fake_core_module = type("CoreModule", (), {"is_compiled_with_cuda": checker})()

        with patch.object(module, "_load_module_without_package_init", return_value=fake_core_module) as load_module_mock:
            result = finder._can_use_paddle_gpu()

        self.assertTrue(result)
        load_module_mock.assert_called_once_with("paddle.base.core")
        checker.assert_called_once_with()

    def test_can_use_paddle_gpu_returns_false_when_core_load_fails(self):
        finder = self.make_finder()

        with patch.object(module, "_load_module_without_package_init", side_effect=ImportError("missing paddle core")) as load_module_mock:
            self.assertFalse(finder._can_use_paddle_gpu())

        load_module_mock.assert_called_once_with("paddle.base.core")

    def test_initialize_paddle_engine_requires_local_models_in_source_mode(self):
        finder = self.make_finder(language="en")

        with patch.object(module, "Image", object()), \
             patch.object(module, "is_packaged_runtime", return_value=False), \
             patch.object(module, "get_local_model_dir", side_effect=[None, None]), \
             patch.object(finder, "error") as error_mock:
            result = finder.initialize_paddle_engine()

        self.assertFalse(result)
        self.assertEqual(error_mock.call_count, 1)
        error_message = error_mock.call_args[0][0]
        # default build_profile is gpu, so det model name should be server_det
        self.assertIn(module.TEXT_DETECTION_MODEL_NAME, error_message)
        self.assertIn("en_PP-OCRv5_mobile_rec", error_message)
        self.assertIn("stays offline", error_message)
        self.assertIn("will not download models at runtime", error_message)

    def test_perform_pdf_ocr_merges_page_text_and_stops_after_match(self):
        pdf_path = self.make_file("source/document.pdf", b"%PDF-1.4")
        finder = self.make_finder(target="你")

        with patch.object(module.OCRFinder, "iter_pdf_page_images", return_value=[(1, "page-1"), (2, "page-2"), (3, "page-3")]), \
             patch.object(module.OCRFinder, "perform_ocr", side_effect=["第一页", "这里有你", "不会执行"]) as perform_ocr_mock:
            result = finder.perform_pdf_ocr(pdf_path)

        self.assertEqual(result, "第一页\n这里有你")
        self.assertEqual(perform_ocr_mock.call_count, 2)

    def test_initialize_backends_passes_selected_language_and_device(self):
        finder = self.make_finder(language="en", device="gpu", build_profile="gpu", backend="paddle")
        fake_ocr_instance = Mock()
        fake_ocr_class = Mock(return_value=fake_ocr_instance)
        det_model_dir = self.root / "cache" / module.TEXT_DETECTION_MODEL_NAME
        rec_model_name = module.get_recognition_model_name("en")
        rec_model_dir = self.root / "cache" / rec_model_name

        with patch.object(module, "Image", object()), \
             patch.object(module, "PaddleOCR", fake_ocr_class), \
             patch.object(module, "get_local_model_dir", side_effect=[det_model_dir, rec_model_dir]):
            result = finder.initialize_backends()

        self.assertTrue(result)
        self.assertIs(finder.ocr_backend, fake_ocr_instance)
        self.assertIs(finder.ocr, fake_ocr_instance)
        self.assertEqual(finder.active_backend_name, "paddle")
        self.assertEqual(finder.resolved_device, "gpu")
        fake_ocr_class.assert_called_once_with(
            lang="en",
            text_detection_model_name=module.TEXT_DETECTION_MODEL_NAME,
            text_detection_model_dir=str(det_model_dir),
            text_recognition_model_name=rec_model_name,
            text_recognition_model_dir=str(rec_model_dir),
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
            device="gpu",
        )

    def test_initialize_backends_forces_cpu_device_for_cpu_build_profile(self):
        finder = self.make_finder(language="en", device="gpu", build_profile="cpu", backend="paddle")
        fake_ocr_instance = Mock()
        fake_ocr_class = Mock(return_value=fake_ocr_instance)
        det_model_dir = self.root / "cache" / module.TEXT_DETECTION_MODEL_NAME_CPU
        rec_model_name = module.get_recognition_model_name("en")
        rec_model_dir = self.root / "cache" / rec_model_name

        with patch.object(module, "Image", object()), \
             patch.object(module, "PaddleOCR", fake_ocr_class), \
             patch.object(module, "get_local_model_dir", side_effect=[det_model_dir, rec_model_dir]):
            result = finder.initialize_backends()

        self.assertTrue(result)
        self.assertEqual(finder.resolved_device, "cpu")
        fake_ocr_class.assert_called_once_with(
            lang="en",
            text_detection_model_name=module.TEXT_DETECTION_MODEL_NAME_CPU,
            text_detection_model_dir=str(det_model_dir),
            text_recognition_model_name=rec_model_name,
            text_recognition_model_dir=str(rec_model_dir),
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
            device="cpu",
        )

    def test_initialize_paddle_engine_aliases_initialize_backends(self):
        finder = self.make_finder()

        with patch.object(module.OCRFinder, "initialize_backends", return_value=True) as initialize_backends_mock:
            result = finder.initialize_paddle_engine()

        self.assertTrue(result)
        initialize_backends_mock.assert_called_once_with()

    def test_check_paddleocr_aliases_initialize_backends(self):
        finder = self.make_finder()

        with patch.object(module.OCRFinder, "initialize_backends", return_value=True) as initialize_backends_mock:
            result = finder.check_paddleocr()

        self.assertTrue(result)
        initialize_backends_mock.assert_called_once_with()

    def test_initialize_paddle_engine_falls_back_to_cpu_in_auto_mode(self):
        finder = self.make_finder(language="en", device="auto")
        fake_ocr_instance = Mock()
        fake_ocr_class = Mock(side_effect=[RuntimeError("gpu init failed"), fake_ocr_instance])
        det_model_dir = self.root / "cache" / module.TEXT_DETECTION_MODEL_NAME
        rec_model_name = module.get_recognition_model_name("en")
        rec_model_dir = self.root / "cache" / rec_model_name

        with patch.object(module, "Image", object()), \
             patch.object(module, "PaddleOCR", fake_ocr_class), \
             patch.object(module, "get_local_model_dir", side_effect=[det_model_dir, rec_model_dir]), \
             patch.object(finder, "_can_use_paddle_gpu", return_value=True):
            result = finder.initialize_paddle_engine()

        self.assertTrue(result)
        self.assertIs(finder.ocr_backend, fake_ocr_instance)
        self.assertIs(finder.ocr, fake_ocr_instance)
        self.assertEqual(finder.active_backend_name, "paddle")
        self.assertEqual(finder.resolved_device, "cpu")

    def test_main_passes_language_device_profile_and_backend_options_to_finder(self):
        fake_finder = Mock()
        fake_finder.run.return_value = 0

        with patch.object(module, "OCRFinder", return_value=fake_finder) as finder_class, \
             patch("sys.argv", [
                 "chinese_ocr_finder.py",
                 "-t", "目标",
                 "-s", "./src",
                 "-o", "./out",
                 "-v",
                 "--language", "en",
                 "--device", "gpu",
                 "--build-profile", "cpu",
                 "--backend", "paddle",
             ]):
            result = module.main()

        self.assertEqual(result, 0)
        finder_class.assert_called_once_with(
            target_chars="目标",
            source_dir="./src",
            output_dir="./out",
            verbose=True,
            language="en",
            device="gpu",
            build_profile="cpu",
            backend="paddle",
        )
        fake_finder.run.assert_called_once_with()

    def test_gui_start_scan_requires_target_text(self):
        gui = object.__new__(gui_module.OCRFinderGUI)
        gui.ui_language = "en"
        gui.worker_thread = None
        gui.target_var = Mock(get=Mock(return_value="   "))
        gui.source_var = Mock(get=Mock(return_value="./src"))
        gui.output_var = Mock(get=Mock(return_value="./out"))
        gui.device_var = Mock(get=Mock(return_value="gpu"))
        gui.backend_var = Mock(get=Mock(return_value="paddle"))
        gui._build_profile = "gpu"

        with patch.object(gui_module.messagebox, "showerror") as showerror_mock:
            gui.start_scan()

        showerror_mock.assert_called_once_with("Missing target", "Please enter the target text to search for.")
        self.assertIsNone(gui.worker_thread)

    def test_gui_start_scan_requires_source_and_output_paths(self):
        cases = (
            ("source", "   ", "./out", ("Missing source", "Please choose a source folder.")),
            ("output", "./src", "   ", ("Missing output", "Please choose an output folder.")),
        )

        for label, source, output, expected_error in cases:
            with self.subTest(missing=label):
                gui = object.__new__(gui_module.OCRFinderGUI)
                gui.ui_language = "en"
                gui.worker_thread = None
                gui.target_var = Mock(get=Mock(return_value="目标"))
                gui.source_var = Mock(get=Mock(return_value=source))
                gui.output_var = Mock(get=Mock(return_value=output))
                gui.device_var = Mock(get=Mock(return_value="gpu"))
                gui.backend_var = Mock(get=Mock(return_value="paddle"))
                gui._build_profile = "gpu"

                with patch.object(gui_module.messagebox, "showerror") as showerror_mock:
                    gui.start_scan()

                showerror_mock.assert_called_once_with(*expected_error)
                self.assertIsNone(gui.worker_thread)

    def test_gui_run_scan_passes_selected_runtime_options_to_finder(self):
        gui = object.__new__(gui_module.OCRFinderGUI)
        gui.ui_language = "en"
        gui.event_queue = queue.Queue()
        gui.stop_event = Mock()
        gui.stop_event.is_set.return_value = False

        fake_finder = Mock()
        fake_finder.run.return_value = 0
        fake_finder.get_summary_data.return_value = {"total_files": 0}

        with patch.object(gui_module, "OCRFinder", return_value=fake_finder) as finder_class:
            gui.run_scan(
                target="目标",
                source="./src",
                output="./out",
                verbose=True,
                language="en",
                device="gpu",
                build_profile="gpu",
                backend="rapidocr",
            )

        finder_class.assert_called_once_with(
            target_chars="目标",
            source_dir="./src",
            output_dir="./out",
            verbose=True,
            language="en",
            device="gpu",
            build_profile="gpu",
            backend="rapidocr",
            event_callback=gui.event_queue.put,
            should_stop=gui.stop_event.is_set,
        )
        worker_event = gui.event_queue.get_nowait()
        self.assertEqual(worker_event["kind"], "worker_finished")
        self.assertEqual(worker_event["exit_code"], 0)

    def test_gui_run_scan_reports_unexpected_errors_to_queue(self):
        gui = object.__new__(gui_module.OCRFinderGUI)
        gui.ui_language = "en"
        gui.event_queue = queue.Queue()
        gui.stop_event = Mock()
        gui.stop_event.is_set.return_value = False

        with patch.object(gui_module, "OCRFinder", side_effect=RuntimeError("boom")):
            gui.run_scan(
                target="目标",
                source="./src",
                output="./out",
                verbose=False,
                language="en",
                device="auto",
                build_profile="gpu",
                backend="paddle",
            )

        error_event = gui.event_queue.get_nowait()
        worker_event = gui.event_queue.get_nowait()
        self.assertEqual(error_event, {"kind": "error", "message": "Unexpected error: boom"})
        self.assertEqual(worker_event["kind"], "worker_finished")
        self.assertEqual(worker_event["exit_code"], 1)
        self.assertIsNone(worker_event["summary"])

    def test_gui_stop_scan_sets_stop_requested_state_for_active_worker(self):
        gui = object.__new__(gui_module.OCRFinderGUI)
        gui.ui_language = "en"
        gui.worker_thread = Mock()
        gui.worker_thread.is_alive.return_value = True
        gui.stop_event = Mock()
        gui.status_var = Mock()
        gui.append_log = Mock()

        gui.stop_scan()

        gui.stop_event.set.assert_called_once_with()
        gui.status_var.set.assert_called_once_with("Stopping after current file...")
        gui.append_log.assert_called_once_with("Stop requested.")

    def test_gui_stop_scan_ignores_missing_or_inactive_worker(self):
        for label, worker_thread in (("missing", None), ("inactive", Mock(is_alive=Mock(return_value=False)))):
            with self.subTest(worker_state=label):
                gui = object.__new__(gui_module.OCRFinderGUI)
                gui.ui_language = "en"
                gui.worker_thread = worker_thread
                gui.stop_event = Mock()
                gui.status_var = Mock()
                gui.append_log = Mock()

                gui.stop_scan()

                gui.stop_event.set.assert_not_called()
                gui.status_var.set.assert_not_called()
                gui.append_log.assert_not_called()

    def test_gui_process_events_drains_queue_and_reschedules_itself(self):
        gui = object.__new__(gui_module.OCRFinderGUI)
        gui.ui_language = "en"
        gui.event_queue = queue.Queue()
        first_event = {"kind": "status", "message": "Starting"}
        second_event = {"kind": "log", "message": "Still running"}
        gui.event_queue.put(first_event)
        gui.event_queue.put(second_event)
        gui.handle_event = Mock()
        gui.after = Mock()

        gui.process_events()

        self.assertEqual(gui.handle_event.call_count, 2)
        self.assertEqual(gui.handle_event.call_args_list[0].args[0], first_event)
        self.assertEqual(gui.handle_event.call_args_list[1].args[0], second_event)
        gui.after.assert_called_once()
        self.assertEqual(gui.after.call_args.args[0], 100)
        self.assertEqual(gui.after.call_args.args[1].__func__, gui_module.OCRFinderGUI.process_events)
        self.assertIs(gui.after.call_args.args[1].__self__, gui)

    def test_gui_handle_event_updates_progress_for_scan_events(self):
        gui = object.__new__(gui_module.OCRFinderGUI)
        gui.ui_language = "en"
        gui.progress = Mock()
        gui.status_var = Mock()
        gui.append_log = Mock()

        gui.handle_event({"kind": "scan_complete", "message": "Found 3 files", "total": 3})

        gui.progress.configure.assert_called_once_with(maximum=3, value=0)
        gui.status_var.set.assert_called_once_with("Found 3 files")
        gui.append_log.assert_called_once_with("Found 3 files")

        gui.progress.configure.reset_mock()
        gui.status_var.set.reset_mock()
        gui.append_log.reset_mock()

        gui.handle_event({"kind": "file_start", "message": "Scanning image.png", "current": 2, "total": 3})

        gui.progress.configure.assert_called_once_with(maximum=3, value=2)
        gui.status_var.set.assert_called_once_with("Scanning image.png")
        gui.append_log.assert_not_called()

    def test_gui_handle_event_logs_error_and_summary_events(self):
        gui = object.__new__(gui_module.OCRFinderGUI)
        gui.ui_language = "en"
        gui.append_log = Mock()
        gui.status_var = Mock()
        gui.format_summary = Mock(return_value="Scanned: 3 | Matches: 1")

        gui.handle_event({"kind": "error", "message": "boom"})

        gui.append_log.assert_called_once_with("ERROR: boom")
        gui.status_var.set.assert_called_once_with("boom")

        gui.append_log.reset_mock()
        gui.status_var.set.reset_mock()

        gui.handle_event({"kind": "cancelled", "message": "Stopped", "summary": {"total_files": 3}})

        self.assertEqual(gui.append_log.call_args_list[0].args[0], "Stopped")
        self.assertEqual(gui.append_log.call_args_list[1].args[0], "Scanned: 3 | Matches: 1")
        gui.status_var.set.assert_called_once_with("Stopped")
        gui.format_summary.assert_called_once_with({"total_files": 3})

        gui.append_log.reset_mock()
        gui.status_var.set.reset_mock()
        gui.format_summary.reset_mock()
        gui.format_summary.return_value = "Scanned: 3 | Matches: 1"

        gui.handle_event({"kind": "finished", "message": "Done", "summary": {"total_files": 3}})

        gui.append_log.assert_called_once_with("Scanned: 3 | Matches: 1")
        gui.status_var.set.assert_called_once_with("Done")
        gui.format_summary.assert_called_once_with({"total_files": 3})

    def test_gui_handle_event_updates_simple_log_and_status_branches(self):
        gui = object.__new__(gui_module.OCRFinderGUI)
        gui.ui_language = "en"
        gui.append_log = Mock()
        gui.status_var = Mock()

        gui.handle_event({"kind": "match", "message": "Matched file.png"})

        gui.append_log.assert_called_once_with("Matched file.png")
        gui.status_var.set.assert_called_once_with("Matched file.png")

        gui.append_log.reset_mock()
        gui.status_var.set.reset_mock()

        gui.handle_event({"kind": "log", "message": "Verbose entry"})

        gui.append_log.assert_called_once_with("Verbose entry")
        gui.status_var.set.assert_not_called()

        gui.append_log.reset_mock()
        gui.status_var.set.reset_mock()

        gui.handle_event({"kind": "progress", "message": "Halfway there"})

        gui.append_log.assert_not_called()
        gui.status_var.set.assert_called_once_with("Halfway there")

        gui.append_log.reset_mock()
        gui.status_var.set.reset_mock()

        gui.handle_event({"kind": "status", "message": "Ready"})

        gui.append_log.assert_called_once_with("Ready")
        gui.status_var.set.assert_called_once_with("Ready")

    def test_gui_handle_event_worker_finished_restores_buttons_and_status(self):
        gui = object.__new__(gui_module.OCRFinderGUI)
        gui.ui_language = "en"
        gui.start_button = Mock()
        gui.stop_button = Mock()
        gui.status_var = Mock()

        for exit_code, expected_status in ((0, "Scan completed."), (2, "Scan cancelled."), (1, "Scan failed.")):
            with self.subTest(exit_code=exit_code):
                gui.start_button.configure.reset_mock()
                gui.stop_button.configure.reset_mock()
                gui.status_var.set.reset_mock()

                gui.handle_event({"kind": "worker_finished", "exit_code": exit_code})

                gui.start_button.configure.assert_called_once_with(state="normal")
                gui.stop_button.configure.assert_called_once_with(state="disabled")
                gui.status_var.set.assert_called_once_with(expected_status)

    def test_gui_on_close_keeps_window_open_when_exit_is_cancelled(self):
        gui = object.__new__(gui_module.OCRFinderGUI)
        gui.ui_language = "en"
        gui.worker_thread = Mock()
        gui.worker_thread.is_alive.return_value = True
        gui.stop_event = Mock()
        gui.destroy = Mock()

        with patch.object(gui_module.messagebox, "askyesno", return_value=False) as askyesno_mock:
            gui.on_close()

        askyesno_mock.assert_called_once_with("Exit", "A scan is still running. Stop and exit?")
        gui.stop_event.set.assert_not_called()
        gui.destroy.assert_not_called()

    def test_gui_on_close_stops_active_worker_before_destroying(self):
        gui = object.__new__(gui_module.OCRFinderGUI)
        gui.ui_language = "en"
        gui.worker_thread = Mock()
        gui.worker_thread.is_alive.return_value = True
        gui.stop_event = Mock()
        gui.destroy = Mock()

        with patch.object(gui_module.messagebox, "askyesno", return_value=True) as askyesno_mock:
            gui.on_close()

        askyesno_mock.assert_called_once_with("Exit", "A scan is still running. Stop and exit?")
        gui.stop_event.set.assert_called_once_with()
        gui.destroy.assert_called_once_with()

    def test_gui_on_close_destroys_immediately_without_active_worker(self):
        for label, worker_thread in (("missing", None), ("inactive", Mock(is_alive=Mock(return_value=False)))):
            with self.subTest(worker_state=label):
                gui = object.__new__(gui_module.OCRFinderGUI)
                gui.ui_language = "en"
                gui.worker_thread = worker_thread
                gui.stop_event = Mock()
                gui.destroy = Mock()

                with patch.object(gui_module.messagebox, "askyesno") as askyesno_mock:
                    gui.on_close()

                askyesno_mock.assert_not_called()
                gui.stop_event.set.assert_not_called()
                gui.destroy.assert_called_once_with()


    def test_gui_build_ui_includes_device_selector(self):
        class FakeWidget:
            def __init__(self, *args, **kwargs):
                self.args = args
                self.kwargs = kwargs

            def grid(self, *args, **kwargs):
                return self

            def columnconfigure(self, *args, **kwargs):
                return None

            def rowconfigure(self, *args, **kwargs):
                return None

        gui = object.__new__(gui_module.OCRFinderGUI)
        gui.ui_language = "en"
        gui.tr = gui_module.OCRFinderGUI.tr.__get__(gui, gui_module.OCRFinderGUI)
        gui.columnconfigure = Mock()
        gui.rowconfigure = Mock()
        gui.target_var = object()
        gui.source_var = object()
        gui.output_var = object()
        gui.device_var = object()
        gui.backend_var = object()
        gui.verbose_var = object()
        gui.status_var = object()
        gui._build_profile = "gpu"
        gui.choose_source = Mock()
        gui.choose_output = Mock()
        gui.start_scan = Mock()
        gui.stop_scan = Mock()
        gui.open_output_folder = Mock()
        gui.clear_log = Mock()

        with patch.object(gui_module.ttk, "Frame", side_effect=FakeWidget), \
             patch.object(gui_module.ttk, "LabelFrame", side_effect=FakeWidget), \
             patch.object(gui_module.ttk, "Label", side_effect=FakeWidget) as label_mock, \
             patch.object(gui_module.ttk, "Entry", side_effect=FakeWidget), \
             patch.object(gui_module.ttk, "Button", side_effect=FakeWidget), \
             patch.object(gui_module.ttk, "Combobox", side_effect=FakeWidget) as combobox_mock, \
             patch.object(gui_module.ttk, "Checkbutton", side_effect=FakeWidget) as checkbutton_mock, \
             patch.object(gui_module.ttk, "Progressbar", side_effect=FakeWidget), \
             patch.object(gui_module, "ScrolledText", side_effect=FakeWidget):
            gui._build_ui()

        label_texts = [call.kwargs.get("text") for call in label_mock.call_args_list]
        self.assertNotIn("Language", label_texts)
        self.assertIn("Device", label_texts)
        self.assertIn("OCR engine", label_texts)
        # gpu build shows both device and backend comboboxes
        self.assertEqual(len(combobox_mock.call_args_list), 2)
        self.assertEqual(combobox_mock.call_args_list[0].kwargs["textvariable"], gui.device_var)
        self.assertEqual(combobox_mock.call_args_list[0].kwargs["values"], gui_module.SUPPORTED_DEVICE_OPTIONS)
        self.assertEqual(combobox_mock.call_args_list[1].kwargs["textvariable"], gui.backend_var)
        self.assertEqual(len(checkbutton_mock.call_args_list), 1)
        self.assertEqual(checkbutton_mock.call_args_list[0].kwargs["text"], "Verbose log")

    def test_gui_start_scan_configures_worker_and_logs_selected_device(self):

        gui = object.__new__(gui_module.OCRFinderGUI)
        gui.ui_language = "en"
        gui.worker_thread = None
        gui.target_var = Mock(get=Mock(return_value=" 目标 "))
        gui.source_var = Mock(get=Mock(return_value=" ./src "))
        gui.output_var = Mock(get=Mock(return_value=" ./out "))
        gui.device_var = Mock(get=Mock(return_value="gpu"))
        gui.backend_var = Mock(get=Mock(return_value="paddle"))
        gui._build_profile = "cpu"
        gui.verbose_var = Mock(get=Mock(return_value=True))
        gui.stop_event = Mock()
        gui.progress = Mock()
        gui.status_var = Mock()
        gui.append_log = Mock()
        gui.start_button = Mock()
        gui.stop_button = Mock()
        gui.run_scan = Mock()

        fake_thread = Mock()

        with patch.object(gui_module.threading, "Thread", return_value=fake_thread) as thread_class:
            gui.start_scan()

        gui.stop_event.clear.assert_called_once_with()
        gui.progress.configure.assert_called_once_with(maximum=1, value=0)
        gui.status_var.set.assert_called_once_with("Starting scan...")
        self.assertEqual(
            [call.args[0] for call in gui.append_log.call_args_list],
            [
                "Target: 目标",
                "Source: ./src",
                "Output: ./out",
                "Device: gpu",
            ],
        )
        gui.start_button.configure.assert_called_once_with(state="disabled")
        gui.stop_button.configure.assert_called_once_with(state="normal")
        thread_class.assert_called_once_with(
            target=gui.run_scan,
            args=("目标", "./src", "./out", True, gui_module.AUTO_OCR_LANGUAGE, "gpu", "cpu", "paddle"),
            daemon=True,
        )
        self.assertIs(gui.worker_thread, fake_thread)
        fake_thread.start.assert_called_once_with()

    def test_gui_start_scan_ignores_request_when_worker_is_alive(self):
        gui = object.__new__(gui_module.OCRFinderGUI)
        gui.ui_language = "en"
        gui.worker_thread = Mock()
        gui.worker_thread.is_alive.return_value = True
        gui.stop_event = Mock()
        gui.progress = Mock()
        gui.status_var = Mock()
        gui.append_log = Mock()
        gui.start_button = Mock()
        gui.stop_button = Mock()

        with patch.object(gui_module.threading, "Thread") as thread_class, \
             patch.object(gui_module.messagebox, "showerror") as showerror_mock:
            gui.start_scan()

        thread_class.assert_not_called()
        showerror_mock.assert_not_called()
        gui.stop_event.clear.assert_not_called()
        gui.progress.configure.assert_not_called()
        gui.status_var.set.assert_not_called()
        gui.append_log.assert_not_called()
        gui.start_button.configure.assert_not_called()
        gui.stop_button.configure.assert_not_called()

    def test_gui_format_summary_includes_elapsed_only_for_numeric_values(self):
        gui = object.__new__(gui_module.OCRFinderGUI)
        gui.ui_language = "en"

        self.assertEqual(
            gui.format_summary(
                {
                    "total_files": 3,
                    "processed_files": 2,
                    "matches_found": 1,
                    "skipped_files": 4,
                    "error_files": 5,
                    "elapsed_time": 1.234,
                }
            ),
            "Scanned: 3 | Processed: 2 | Matches: 1 | Skipped: 4 | Errors: 5 | Elapsed: 1.23s",
        )
        self.assertEqual(
            gui.format_summary({"total_files": 1, "elapsed_time": "n/a"}),
            "Scanned: 1 | Processed: 0 | Matches: 0 | Skipped: 0 | Errors: 0",
        )

    def test_gui_choose_source_updates_selected_directory(self):
        gui = object.__new__(gui_module.OCRFinderGUI)
        gui.ui_language = "en"
        gui.source_var = Mock()
        gui.source_var.get.return_value = str(self.root / "source")

        with patch.object(gui_module.filedialog, "askdirectory", return_value=str(self.root / "picked-source")) as askdirectory_mock:
            gui.choose_source()

        askdirectory_mock.assert_called_once_with(initialdir=str(self.root / "source"))
        gui.source_var.set.assert_called_once_with(str(self.root / "picked-source"))

    def test_gui_choose_output_uses_cwd_fallback_and_ignores_cancelled_dialog(self):
        gui = object.__new__(gui_module.OCRFinderGUI)
        gui.ui_language = "en"
        gui.output_var = Mock()
        gui.output_var.get.return_value = ""

        with patch.object(gui_module.filedialog, "askdirectory", return_value="") as askdirectory_mock:
            gui.choose_output()

        askdirectory_mock.assert_called_once_with(initialdir=str(Path.cwd()))
        gui.output_var.set.assert_not_called()

    def test_gui_log_helpers_update_text_widget_state_and_contents(self):
        gui = object.__new__(gui_module.OCRFinderGUI)
        gui.ui_language = "en"
        gui.log_text = Mock()

        gui.append_log("hello")

        self.assertEqual(gui.log_text.configure.call_args_list[0].kwargs, {"state": "normal"})
        self.assertEqual(gui.log_text.configure.call_args_list[1].kwargs, {"state": "disabled"})
        gui.log_text.insert.assert_called_once_with("end", "hello\n")
        gui.log_text.see.assert_called_once_with("end")

        gui.log_text.reset_mock()

        gui.clear_log()

        self.assertEqual(gui.log_text.configure.call_args_list[0].kwargs, {"state": "normal"})
        self.assertEqual(gui.log_text.configure.call_args_list[1].kwargs, {"state": "disabled"})
        gui.log_text.delete.assert_called_once_with("1.0", "end")

    def test_gui_main_instantiates_app_and_runs_mainloop(self):
        app = Mock()

        with patch.object(gui_module, "OCRFinderGUI", return_value=app) as app_class:
            gui_module.main()

        app_class.assert_called_once_with()
        app.mainloop.assert_called_once_with()

    def test_gui_open_output_folder_creates_directory_and_uses_windows_launcher(self):
        gui = object.__new__(gui_module.OCRFinderGUI)
        gui.ui_language = "en"
        output_dir = self.root / "nested" / "output"
        gui.output_var = Mock(get=Mock(return_value=str(output_dir)))

        with patch.object(gui_module.os, "name", "nt"), patch.object(
            gui_module.os, "startfile", create=True
        ) as startfile_mock:
            gui.open_output_folder()

        self.assertTrue(output_dir.is_dir())
        startfile_mock.assert_called_once_with(output_dir.resolve())

    def test_gui_open_output_folder_reports_launcher_errors(self):
        gui = object.__new__(gui_module.OCRFinderGUI)
        gui.ui_language = "en"
        output_dir = self.root / "output"
        gui.output_var = Mock(get=Mock(return_value=str(output_dir)))

        with patch.object(gui_module.os, "name", "nt"), patch.object(
            gui_module.os, "startfile", side_effect=OSError("boom"), create=True
        ), patch.object(gui_module.messagebox, "showerror") as showerror_mock:
            gui.open_output_folder()

        self.assertTrue(output_dir.is_dir())
        showerror_mock.assert_called_once_with("Open output failed", "boom")


if __name__ == "__main__":
    unittest.main()
