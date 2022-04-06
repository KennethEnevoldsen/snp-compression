"""
CLI:
CLI for converting .plink format to .zarr format
python src/cli --bfile <plink_file> --out <output_file>.zarr
should include an is working symbol and time (see below)
"""

import time
from typing import Optional
from yaspin import yaspin
from yaspin.spinners import Spinners

from prompt_toolkit import HTML, print_formatted_text
from prompt_toolkit.styles import Style

# override print with feature-rich ``print_formatted_text`` from prompt_toolkit
print = print_formatted_text

# build a basic prompt_toolkit style for styling the HTML wrapped text


def sp_color_print(sp, text={"msg": "", "sub-msg": ""}, style: Optional[dict] = None):
    """custom print function for spinner"""

    if style:
        style = Style.from_dict(style)
    else:
        style = Style.from_dict({
            'msg': '#4caf50 bold',
            'sub-msg': '#616161 italic'
        })
    with sp.hidden():
        print(HTML(
            f'<b>></b> <msg>{text["msg"]}</msg> <sub-msg>{text["sub-msg"]}</sub-msg>'
        ), style=style)


with yaspin(Spinners.arc, text="Running", timer=True, color="cyan") as sp:
    sp.text = "Loading..."
    time.sleep(3.1415)
    sp_color_print(sp, {"msg": "Dataset", "sub-msg": "Loading complete"})
    sp.text = "Writing..."
    time.sleep(3.1415)
    sp_color_print(sp, {"msg": "Dataset", "sub-msg": "Written to disk"})
    sp.text = "Done"
    time.sleep(3.1415)
    sp.ok("✔")
