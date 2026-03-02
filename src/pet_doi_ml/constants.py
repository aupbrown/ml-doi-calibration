"""Physical constants and hardware parameters for the CZT PET detector system.

All magic numbers live here. Every other module imports from this file.
"""

# --- Digitizer / waveform parameters ---
NUM_CHANNELS: int = 16
SAMPLES_PER_WAVEFORM: int = 2001  # slit-scan acquisition mode
PRE_TRIGGER_SAMPLES: int = 200    # first 3.2 µs used as baseline region
SAMPLE_RATE_HZ: float = 62.5e6   # CAEN DT5730 sample rate
DT_S: float = 1.0 / SAMPLE_RATE_HZ  # 16 ns per sample

# --- Motor scan parameters ---
MOTOR_STEP_MM: float = 8.0   # physical step size between collimator positions
NUM_MOTOR_STEPS: int = 41    # total number of scan positions

# --- RENA-3 ASIC emulation parameters ---
RENA3_SHAPING_TIME_S: float = 2.8e-6  # RC-CR peaking time
RENA3_ADC_BITS: int = 12
RENA3_ADC_MAX: int = (1 << RENA3_ADC_BITS) - 1  # 4095

# --- Channel mapping (from view_events.py / CAEN digitizer channel order) ---
# Channels 0-7: anode strips (channel 3 is the steering electrode, not an anode)
# Channels 8-15: cathode strips
CHANNEL_LABELS: tuple[str, ...] = (
    "A18", "A19", "A20", "STR",  # channels 0-3
    "A16", "A17", "A21", "A22",  # channels 4-7
    "C4",  "C5",  "C3",  "C1",  # channels 8-11
    "C2",  "C6",  "C7",  "C8",  # channels 12-15
)

STEERING_CHANNEL: int = 3

# Anode channels: all channels 0-7 except the steering electrode
ANODE_INDICES: tuple[int, ...] = (0, 1, 2, 4, 5, 6, 7)

# Cathode channels: all channels 8-15
CATHODE_INDICES: tuple[int, ...] = (8, 9, 10, 11, 12, 13, 14, 15)
