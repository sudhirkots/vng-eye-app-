# Balance Band firmware (Seeed XIAO nRF52840 Sense)

**Status: written, not yet compiled or run on a board.** Expect small fixes on first build.

What it does: reads the board's built-in motion sensor (LSM6DS3TR-C) **100 times a second** and streams
every reading over **Bluetooth LE** as device **"BalanceBand"**. The recording app (`../recorder/`) connects
to it from Chrome.

## Put it on the board (Arduino IDE)
1. Install the **Arduino IDE** (2.x).
2. *File → Preferences → Additional boards manager URLs*, add
   `https://files.seeedstudio.com/arduino/package_seeeduino_boards_index.json`
3. *Tools → Board → Boards Manager*: install **"Seeed nRF52 Boards"** (the plain one, **not** "mbed-enabled";
   this firmware uses the Bluefruit Bluetooth library that comes with it).
4. *Tools → Manage Libraries*: install **"Seeed Arduino LSM6DS3"**.
5. Open `balance_band_fw/balance_band_fw.ino`, choose board **"Seeed XIAO nRF52840 Sense"** and its USB port,
   and click **Upload**.

## Lights
| LED | Meaning |
|-----|---------|
| Green blinking | On, waiting for the app to connect |
| Blue | Connected and streaming |
| Red blinking | Motion sensor not found (check the board is the **Sense** version) |

## Data format (for anyone changing the app)
Nordic UART Service `6E400001-B5A3-F393-E0A9-E50E24DCCA9E`, notifications on TX `6E400003-…`.
Each sample is a 20-byte frame, little-endian:

| Bytes | Field |
|-------|-------|
| 0–1 | sync `0xA5 0x5A` |
| 2 | sequence number 0–255 (lets the app count lost samples) |
| 3–6 | time stamp, microseconds (u32, wraps every ~71 min; the app unwraps it) |
| 7–12 | accelerometer X, Y, Z, int16, **milli-g** |
| 13–18 | gyroscope X, Y, Z, int16, **0.01 °/s** |
| 19 | checksum = sum of bytes 2–18, & 0xFF |

Frames go out in batches of 10. The app treats the data as a byte stream, so packet splitting does not matter.

## Not in this first version
- Battery level reporting.
- Sleep / power saving when not connected (use the slide switch).
- Recording to the board's memory without a connection.
