from dataclasses import dataclass, fields, make_dataclass

import serial
from mcp.server.fastmcp import FastMCP

LED_MAX_BRIGHTNESS = 32
BRIGHTNESS_DIVIDER = 2


@dataclass
class Color:
  r: int
  g: int
  b: int


fields_defs = [(f"v{i}", Color) for i in range(64)]
LedMatrixInput = make_dataclass("LedMatrixInput", fields_defs)


mcp = FastMCP("emotional_led_matrix")


@mcp.tool()
# async def emotional_led_matrix(values: list[int]) -> str:
async def emotional_led_matrix(input: LedMatrixInput) -> str:
  """
  You have 8x8, 64 total RGB full color LED matrix.
  With this LED you can express emotions, write letters and draw pictures.

  The LEDs are lined up as follows.

  v0,  v1,  v2,  v3,  v4,  v5,  v6,  v7 \n
  v8,  v9,  v10, v11, v12, v13, v14, v15\n
  v16, v17, v18, v19, v20, v21, v22, v23\n
  v24, v25, v26, v27, v28, v29, v30, v31\n
  v32, v33, v34, v35, v36, v37, v38, v39\n
  v40, v41, v42, v43, v44, v45, v46, v47\n
  v48, v49, v50, v51, v52, v53, v54, v55\n
  v56, v57, v58, v59, v60, v61, v62, v63\n

  Args:
    The LedMatrixInput class has 64 Color classes, each with a r(red), g(green), and b(blue) field,
    corresponding to the R, G, and B values of a single LED.
    The range of r, g, b values is as follows.

    r(red): 0 ~ 255
    g(green): 0 ~ 255
    b(blue): 0 ~ 255

  Returns:
    Result of command execution.
  """
  ser = serial.Serial("/dev/ttyUSB0", 115200)
  rgb_data: list[int] = []
  for f in fields(input):
    value = getattr(input, f.name)
    r = int(value.r // BRIGHTNESS_DIVIDER)
    g = int(value.g // BRIGHTNESS_DIVIDER)
    b = int(value.b // BRIGHTNESS_DIVIDER)
    rgb_data.extend([r, g, b])
  ser.write(bytearray(rgb_data))
  ser.close()

  return "Success!!"


if __name__ == "__main__":
  mcp.run(transport="stdio")
  # import asyncio

  # color = Color(10, 0, 0)
  # colors = [color] * 64
  # input = LedMatrixInput(*colors)

  # asyncio.run(emotional_led_matrix(input))
