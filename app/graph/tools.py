"""Tool wrapper functions for SheetAgent action executors.

This module provides LangChain tool wrappers for the SheetAgent action executors.
Each function is decorated with @tool and provides a clear interface for the LLM
to interact with the underlying ActionExecutor classes.
"""
import logging
import re
from typing import Annotated
from langchain_core.tools import tool, InjectedToolCallId
from langchain_core.messages import ToolMessage
from langgraph.prebuilt import InjectedState
from langgraph.types import Command
from pathlib import Path

from app.graph.state import GraphState
from app.core.actions import PythonInterpreter, SheetSelector, AnswerSubmitter
from app.utils.types import TableRepType

logger = logging.getLogger(__name__)

@tool("python_executor", parse_docstring=True)
def python_executor(
    code: str,
    tool_call_id: Annotated[str, InjectedToolCallId],
    state: Annotated[GraphState, InjectedState],
) -> dict:
    """Execute Python code in a sandboxed environment.

    This tool allows you to execute Python code to analyze, manipulate, and visualize
    data in the workbook.

    Args:
        code: The Python code to execute.
    """
    sandbox = state["sandbox"]
    executor = PythonInterpreter(sandbox)
    previous_sheet_state = state["current_sheet_state"]
    response = executor.utilize(code)
    current_sheet_state = sandbox.get_sheet_state()

    return Command(
        update={
            "current_sheet_state": current_sheet_state,
            "previous_sheet_state": previous_sheet_state,
            "messages": [
                ToolMessage(
                    content=response.obs,
                    tool_call_id=tool_call_id,
                    name="python_executor",
                )
            ],
        }
    )

@tool(name_or_callable="cell_range_reader")
def cell_range_reader(
    sheet_name: str,
    cell_range: str,
    state: Annotated[GraphState, InjectedState]
) -> str:
    """Read the values from a specific sheet and cell range.
    
    This tool allows you to read the values of cells from a specific sheet and cell range.
    
    Args:
        sheet_name: The name of the sheet to read
        cell_range: The cell range to read (e.g. "A1:B2")
        state: The current state of the graph (automatically provided)
    
    Returns:
        The data from the sheet and cell range
    """
    # Check that the range is valid
    if not cell_range:
        return "Error: Cell range is empty"
    
    if not sheet_name:
        return "Error: Sheet name is empty"
    
    # Check the format to be A1:B2
    if not re.match(r'^[A-Z]+\d+:[A-Z]+\d+$', cell_range):
        return "Error: Invalid cell range format. Provide a valid cell range in the format A1:B2."
        
    sandbox = state["sandbox"]
    executor = PythonInterpreter(sandbox)
    code = f"""
sheet = workbook["{sheet_name}"]
rows = sheet["{cell_range}"]
if len(rows) == 0:
    print("Error: The cell range is empty. Please provide a valid cell range.")
    return

if len(rows) > 10:
    print("Error: The cell range is too large. Please provide a cell range with a maximum of 10 rows and 10 columns.")
    return

if len(rows[0]) > 10:
    print("Error: The cell range is too large. Please provide a cell range with a maximum of 10 rows and 10 columns.")
    return

for row in sheet["{cell_range}"]:
    for cell in row:
        print(cell.row, cell.column, cell.value)
"""

    response = executor.utilize(code)
    return response.obs


@tool(name_or_callable="get_cumulative_rows", parse_docstring=True)
def get_cumulative_rows(
        sheet_name: str,
        column: int,
        regex_pattern: str,
        state: Annotated[GraphState, InjectedState]
) -> str:
    """Identifies cumulative rows by searching for a pattern in a specific column.

    This tool iterates through each cell in a given column and uses a regular expression
    to determine if the row should be considered "cumulative." Use this after inspecting
    the sheet to decide which column and pattern are best for identifying these rows.

    Args:
        sheet_name: The name of the sheet to analyze.
        column: The 1-based index of the column to search within (e.g., 1 for column 'A', 2 for 'B').
        regex_pattern: The regular expression pattern to match against cell values.
                       The search is case-insensitive.

    Returns:
        A string containing the list of detected cumulative row numbers.
    """
    if not column:
        return "Error: The 'column' argument is missing."

    if not regex_pattern:
        return "Error: The 'regex_pattern' argument is missing."

    if not sheet_name:
        return "Error: The 'sheet_name' argument is missing."

    sandbox = state["sandbox"]
    executor = PythonInterpreter(sandbox)

    # Note: The column index is adjusted to be 0-based for pandas/openpyxl access.
    code = f"""
import re

sheet = workbook["{sheet_name}"]
cumulative_rows = []

# Using iter_rows to go through the sheet row by row
for i, row in enumerate(sheet.iter_rows(min_row=1), 1):
    # Adjust for 0-based index and ensure the column exists
    col_index = {column} - 1
    if col_index < len(row):
        cell_value = str(row[col_index].value)
        # Check if the cell has a value before searching
        if cell_value and re.search(r'{regex_pattern}', cell_value, re.IGNORECASE):
            cumulative_rows.append(i)

print(f"The following rows have been detected as cumulative: {{cumulative_rows}}")
"""

    response = executor.utilize(code)
    return response.obs
