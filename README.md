# xyz2pov

This Python script converts xyz coordinate files into povray scenes.

So far, only hydrogen and carbon atoms are supported, but this is extendable for
other atoms. The style is similar to Avogadro's tube model.
The camera is automatically directed to the molecule's center of mass and placed
at an appropriate distance. Two lights are placed relative to the camera
automatically.

TODO: automatic rotation of the camera to get a horizontal view of the
molecule.
