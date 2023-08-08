Getting Started
===============

Installation
------------

To use Vpower, you need to first install `ANN <http://www.cs.umd.edu/~mount/ANN/>`_ and `Voxelize <https://github.com/leanderthiele/voxelize>`_. If you do not need the Voxelize interpolator, you can choose to install ANN only.

.. Complete installation guide later. Do the core documentation first.

Quickstart
-----------

In this section you will be able to learn how to load a snapshot, interpolate the density and velocity field, and compute the first velocity power spectrum with Vpower.

.. code-block:: python

    import vpower.interp, vpower.spctrm
    snapshot = "/dir/your_snapshot.hdf5"
    simParticles = vpower.interp.load_snapshot(snapshot)

The :code:`load_snapshot` function loads the mass, density, and velocity of particles from the snapshot into a :code:`SimulationParticles` object. Now that you have the data, it is time to interpolate. With Vpower, interpolation is as simple as

.. code-block:: python

    simField = simParticles.interp_to_field(Nsize=256)

Now that the mass, density, and velocity of the particles is interpolated to :math:`256^3` fields in a :code:`SimulationField3D` object. The mass, density, and velocity are interpolated at the same time because the interpolation process considers a mass-weighted average to calculate the velocity field. Therefore it is impossible to interpolate velocity alone without mass or density. (And since all pixels have equal volume so mass and density are basically the same thing except for a constant) 

You can now take a quick peek of the result

.. code-block:: python

    import matplotlib.pyplot as plt
    plt, ax = plt.subplots(1, 2, figsize=(15, 6))
    simField.plot_density_field(index=128, axis=2, ax=ax[0])
    simField.plot_velocity_field(component=0, index=128, axis=2, ax=ax[1])
    plt.show()

To use Voxelize interpolator, simply run the following lines instead

.. code-block:: python

    import vpower.voxelize
    simField = simParticles.voxelize_interp_to_field(Nsize=256)

You can try running both and compare the results.









