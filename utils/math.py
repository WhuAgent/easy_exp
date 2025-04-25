import re
from typing import List, Optional, Tuple


# Regular expression for finding a code block
CODE_BLOCK_PATTERN = r"```(\w*)\n(.*?)\n```"
UNKNOWN = "unknown"


def extract_code(
    text: str, pattern: str = CODE_BLOCK_PATTERN, detect_single_line_code: bool = False
) -> List[Tuple[str, str]]:
    """Extract code from a text.

    Args:
        text (str or List): The content to extract code from. The content can be
            a string or a list, as returned by standard GPT or multimodal GPT.
        pattern (str, optional): The regular expression pattern for finding the
            code block. Defaults to CODE_BLOCK_PATTERN.
        detect_single_line_code (bool, optional): Enable the new feature for
            extracting single line code. Defaults to False.

    Returns:
        list: A list of tuples, each containing the language and the code.
          If there is no code block in the input text, the language would be "unknown".
          If there is code block but the language is not specified, the language would be "".
    """
    # text = content_str(text)
    if not detect_single_line_code:
        match = re.findall(pattern, text, flags=re.DOTALL)
        return match if match else [(UNKNOWN, text)]

    # Extract both multi-line and single-line code block, separated by the | operator
    # `{3}(\w+)?\s*([\s\S]*?)`{3}: Matches multi-line code blocks.
    #    The (\w+)? matches the language, where the ? indicates it is optional.
    # `([^`]+)`: Matches inline code.
    code_pattern = re.compile(r"`{3}(\w+)?\s*([\s\S]*?)`{3}|`([^`]+)`")
    code_blocks = code_pattern.findall(text)

    # Extract the individual code blocks and languages from the matched groups
    extracted = []
    for lang, group1, group2 in code_blocks:
        if group1:
            extracted.append((lang.strip(), group1.strip()))
        elif group2:
            extracted.append(("", group2.strip()))

    return extracted


def remove_boxed(string: str) -> Optional[str]:
    """Source: https://github.com/hendrycks/math
    Extract the text within a \\boxed{...} environment.
    Example:

    > remove_boxed("\\boxed{\\frac{2}{3}}")

    \\frac{2}{3}
    """
    left = "\\boxed{"
    try:
        if not all((string[: len(left)] == left, string[-1] == "}")):
            raise AssertionError

        return string[len(left) : -1]
    except Exception:
        return None


def last_boxed_only_string(string: str) -> Optional[str]:
    """Source: https://github.com/hendrycks/math
    Extract the last \\boxed{...} or \\fbox{...} element from a string.
    """
    idx = string.rfind("\\boxed")
    if idx < 0:
        idx = string.rfind("\\fbox")
        if idx < 0:
            return None

    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1

    if right_brace_idx is None:
        retval = None
    else:
        retval = string[idx : right_brace_idx + 1]

    return retval


def get_answer(solution: Optional[str]) -> Optional[str]:
    if solution is None:
        return None
    last_boxed = last_boxed_only_string(solution)
    if last_boxed is None:
        return None
    answer = remove_boxed(last_boxed)
    if answer is None:
        return None
    return answer
