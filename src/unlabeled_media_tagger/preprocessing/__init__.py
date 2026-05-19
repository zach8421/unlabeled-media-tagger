"""Preprocessing utilities shared between the local pipeline and the Colab
face-preprocessing notebook.

Functions here are kept dependency-light (cv2, numpy only) and side-effect free
so they can be lifted into the Colab notebook's helpers cell verbatim when the
notebook hits v1.1.
"""
