from dataclasses import dataclass, fields, make_dataclass

import serial
from mcp.server.fastmcp import FastMCP

LED_MAX_BRIGHTNESS = 32
BRIGHTNESS_DIVIDER = 8


@dataclass
class Color:
  red: int
  green: int
  blue: int


fields_defs = [(f"led_{i}", Color) for i in range(64)]
LedMatrixInput = make_dataclass("LedMatrixInput", fields_defs)


mcp = FastMCP("emotional_led_matrix")


@mcp.tool()
# async def emotional_led_matrix(values: list[int]) -> str:
async def emotional_led_matrix(input: LedMatrixInput) -> str:
  """
  You have 8x8, 64 total RGB full color LED matrix.
  With this LED you can express emotions, write letters and draw pictures.

  Args:
    The LedMatrixInput class has 64 Color classes, each with a red, green, and blue field,
    corresponding to the R, G, and B values of a single LED.
    The range of red, gree, blue values is as follows.

    red: 0 ~ 255
    green: 0 ~ 255
    blue: 0 ~ 255

  Returns:
    Result of command execution.
  """
  ser = serial.Serial("/dev/ttyUSB0", 115200)
  rgb_data: list[int] = []
  for f in fields(input):
    value = getattr(input, f.name)
    r = int(value.red // BRIGHTNESS_DIVIDER)
    g = int(value.green // BRIGHTNESS_DIVIDER)
    b = int(value.blue // BRIGHTNESS_DIVIDER)
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
