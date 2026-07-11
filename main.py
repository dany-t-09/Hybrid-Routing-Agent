import argparse
import ctypes
import os
import sys
import time
from pathlib import Path
from ctypes import wintypes

from batch_runner import default_input_path, default_output_path, process_tasks
from classifier import classify_task
from solver import solve


_IGNORED_PASTE_REMAINDER = object()
_pending_paste_lines: list[str] = []


def _get_clipboard_text() -> str | None:
    """Read Unicode clipboard text on Windows, returning None when unavailable.

    Clipboard text is backed by a movable global-memory allocation.  Reading it
    with ``wstring_at(pointer)`` makes ctypes scan for a terminating NUL, which
    can run past a malformed or changing clipboard allocation and crash the
    process.  Read the allocation's exact size instead.
    """
    if os.name != "nt":
        return None

    user32 = ctypes.windll.user32
    kernel32 = ctypes.windll.kernel32
    user32.OpenClipboard.argtypes = [wintypes.HWND]
    user32.OpenClipboard.restype = wintypes.BOOL
    user32.GetClipboardData.restype = ctypes.c_void_p
    user32.CloseClipboard.restype = wintypes.BOOL
    kernel32.GlobalLock.argtypes = [ctypes.c_void_p]
    kernel32.GlobalLock.restype = ctypes.c_void_p
    kernel32.GlobalSize.argtypes = [ctypes.c_void_p]
    kernel32.GlobalSize.restype = ctypes.c_size_t
    kernel32.GlobalUnlock.argtypes = [ctypes.c_void_p]
    kernel32.GlobalUnlock.restype = wintypes.BOOL
    if not user32.OpenClipboard(None):
        return None
    try:
        clipboard_handle = user32.GetClipboardData(13)  # CF_UNICODETEXT
        if not clipboard_handle:
            return None
        byte_count = kernel32.GlobalSize(clipboard_handle)
        if byte_count < 2:
            return None
        text_pointer = kernel32.GlobalLock(clipboard_handle)
        if not text_pointer:
            return None
        try:
            raw_text = ctypes.string_at(text_pointer, byte_count)
            return raw_text.decode("utf-16-le", errors="replace").split("\0", 1)[0]
        except (OSError, ValueError, ctypes.ArgumentError):
            return None
        finally:
            kernel32.GlobalUnlock(clipboard_handle)
    finally:
        user32.CloseClipboard()


def _clear_pending_console_input() -> None:
    """Remove remaining lines injected by a multiline clipboard paste on Windows."""
    if os.name != "nt":
        return
    try:
        import msvcrt

        console_handle = msvcrt.get_osfhandle(sys.stdin.fileno())
        ctypes.windll.kernel32.FlushConsoleInputBuffer(console_handle)
    except (OSError, ValueError):
        pass


def _read_queued_paste_text() -> str:
    """Return text already queued behind the first pasted console line.

    Some terminals do not expose pasted text through the Windows clipboard API.
    After ``input()`` accepts the first line, the rest of a multiline paste is
    still in the console input buffer; consume it directly before solving.
    """
    if os.name != "nt":
        return ""

    try:
        import msvcrt

        characters: list[str] = []
        deadline = time.monotonic() + 0.15
        while time.monotonic() < deadline:
            if msvcrt.kbhit():
                characters.append(msvcrt.getwch())
                # Keep a short grace period while the paste is still arriving.
                deadline = time.monotonic() + 0.03
            else:
                time.sleep(0.005)
        return "".join(characters)
    except (OSError, ValueError):
        return ""


def _complete_pasted_query(first_line: str, clipboard_text: str | None) -> str:
    """Use the full clipboard only when it matches the first line just pasted."""
    if not clipboard_text or "\n" not in clipboard_text:
        return first_line.strip()

    pasted_query = clipboard_text.rstrip("\r\n")
    pasted_first_line = pasted_query.splitlines()[0].strip() if pasted_query.splitlines() else ""
    if pasted_first_line != first_line.strip():
        return first_line.strip()

    global _pending_paste_lines
    _pending_paste_lines = pasted_query.splitlines()[1:]
    _clear_pending_console_input()
    return pasted_query.strip()


def read_query() -> str | None | object:
    """Read one Enter-terminated query, recovering complete multiline clipboard pastes."""
    global _pending_paste_lines
    try:
        first_line = input("Ask the general agent (/exit quits): ")
    except EOFError:
        return None

    while _pending_paste_lines and first_line == _pending_paste_lines[0]:
        _pending_paste_lines.pop(0)
        if not _pending_paste_lines:
            return _IGNORED_PASTE_REMAINDER
        try:
            first_line = input()
        except EOFError:
            return None

    # A different line means the console buffer was cleared and this is the
    # user's next genuine query, not a delayed paste remainder.
    _pending_paste_lines = []
    query = _complete_pasted_query(first_line, _get_clipboard_text())
    if query != first_line.strip():
        return query

    queued_paste_text = _read_queued_paste_text()
    if queued_paste_text:
        # ``input()`` consumed the newline after the first line.
        return f"{first_line}\n{queued_paste_text}".strip()
    return query


def run_interactive() -> None:
    while True:
        query = read_query()
        if query is _IGNORED_PASTE_REMAINDER:
            continue
        if query is None or query.lower() in {"/exit", "/quit", "exit", "quit"}:
            print("Goodbye.")
            return
        if not query:
            print("Please enter a question or task.\n")
            continue

        task_type = classify_task(query)
        result = solve(query, task_type)
        print(f"\n{result.answer}")
        accuracy = (
            f"Estimated answer accuracy: {result.estimated_accuracy}%"
            if result.estimated_accuracy is not None
            else "Estimated answer accuracy: unavailable"
        )
        print(accuracy)
        print(f"Answered by: {result.source}")
        if result.source == "Fireworks AI":
            if result.total_tokens is None:
                print("Fireworks tokens used: unavailable (not reported by API)")
            else:
                print(f"Fireworks tokens used: {result.total_tokens}")
        print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the general agent interactively or on a JSON task batch.")
    parser.add_argument("--batch", action="store_true", help="Read tasks JSON and write results JSON, then exit.")
    parser.add_argument("--input", type=Path, default=None, help="Path to tasks.json (used with --batch).")
    parser.add_argument("--output", type=Path, default=None, help="Path to results.json (used with --batch).")
    args = parser.parse_args()

    if not args.batch:
        run_interactive()
        return

    input_path = args.input or Path(os.getenv("TASKS_INPUT_PATH", str(default_input_path())))
    output_path = args.output or Path(os.getenv("TASKS_OUTPUT_PATH", str(default_output_path())))
    try:
        results = process_tasks(input_path, output_path)
    except ValueError as exc:
        parser.error(str(exc))
    print(f"Wrote {len(results)} result(s) to {output_path}")


if __name__ == "__main__":
    main()
