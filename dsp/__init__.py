"""Audio DSP: 6-channel sweet-spot steering (delays, levels, off-axis treble) and playback.

engine.SteeringEngine is the per-block processor; player.Player runs it on a sound
device driven by the vision bridge; virtual.VirtualSurround renders the 5.1 room over
stereo for testing without speakers. Run with tools/dsp.py (the app starts it).
"""
