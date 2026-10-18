# Third-party notices

`src/abstract_harness/robot/core`, `src/abstract_harness/robot/agent` and
`src/abstract_harness/robot/yam`, with their tests in `tests/robot_core`, `tests/robot_agent`
and `tests/robot_yam`, are derived from MIT-licensed code: the core and agent from
github.com/robocurve/inspect-robots at commit 095172fb8452ac1d47da49834df4d3faa562e4ee
(release 0.60.0, agent 0.28.0), and the YAM components from
github.com/robocurve/inspect-robots-yam at commit 3b1e0856671e19bfc575e74a8c8efba03ca5b242
(release 0.37.0). They were renamed and changed here.

Bundled robot models, also MIT-licensed:

- `src/abstract_harness/robot/yam/assets/yam_collision.xml`: a collision-only model
  generated from MuJoCo Menagerie's `i2rt_yam/yam.xml` (github.com/google-deepmind/mujoco_menagerie
  at commit 71f066ad0be9cd271f7ed58c030243ef157af9f4), copyright i2rt robotics. The license is
  in `MENAGERIE_LICENSE` beside it.

Dependencies installed from PyPI or git (MuJoCo, i2rt, Rerun, librealsense, OpenCV and the
others in `pyproject.toml`) are not redistributed here and keep their own licenses.

## robocurve/inspect-robots

```
MIT License

Copyright (c) 2026 robocurve

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## robocurve/inspect-robots-yam

```
MIT License

Copyright (c) 2026 RoboCurve

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```
