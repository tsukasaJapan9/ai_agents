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

  The LEDs are lined up as follows.

  led_0,  led_1,  led_2,  led_3,  led_4,  led_5,  led_6,  led_7 \n
  led_8,  led_9,  led_10, led_11, led_12, led_13, led_14, led_15\n
  led_16, led_17, led_18, led_19, led_20, led_21, led_22, led_23\n
  led_24, led_25, led_26, led_27, led_28, led_29, led_30, led_31\n
  led_32, led_33, led_34, led_35, led_36, led_37, led_38, led_39\n
  led_40, led_41, led_42, led_43, led_44, led_45, led_46, led_47\n
  led_48, led_49, led_50, led_51, led_52, led_53, led_54, led_55\n
  led_56, led_57, led_58, led_59, led_60, led_61, led_62, led_63\n

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
