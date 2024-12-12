
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_mgga_c_tpssloc_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_tpssloc", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([-4.381808782076816e-02, -3.904981811851881e-02, -2.154785473145320e-04, -2.636037200048630e-03, -2.967849098698696e-09, -5.097572301317905e-13, -1.609362599709138e-18])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_mgga_c_tpssloc_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_tpssloc", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([-1.094159027920180e-01, -1.094203713315945e-01, -9.353809529757991e-02, -9.352394382565268e-02, -2.113520070253435e-03, -2.117604242135843e-03, -3.038818803404198e-02, -4.686838237826788e-01, -1.309869198233306e-03, -4.579227113375993e-03, -5.642883504078990e-12, -5.711353446146106e-12, -6.679208219366847e-19, -2.012837647485330e-18])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_c_tpssloc_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_tpssloc", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = numpy.asarray([1.252690213666526e-04, 2.902753707133360e-04, 1.253459558919385e-04, 2.686035709051684e-04, 5.935071851718008e-04, 2.686900344949031e-04, 1.039784351927314e-03, -3.291085052841001e-04, 1.041836247124721e-03, 3.440299554280611e+01, 8.690701408631882e+01, 1.103419273840713e+03, 7.643367559456637e+00, 1.527683289847397e+01, 2.906859251020732e+04, 3.564298828446929e-08, 1.791908953434545e-08, 3.623443211502231e-08, -8.404585812582010e-10, -1.680917162472331e-09, -8.404585812730616e-10])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_c_tpssloc_Li_2_vtau():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_tpssloc", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vtau"].flatten()
    ref_tgt = numpy.asarray([-1.335702424797275e-02, -1.335702424797275e-02, -7.960833830631836e-03, -7.960833830631835e-03, 3.052429734968461e-05, 3.052429734968458e-05, -1.191993114741734e+00, -1.191993114741472e+00, -1.827878883982652e-02, -1.827878882505897e-02, 2.774616267609913e-18, 2.774616267609915e-18, -1.123538437260015e-33, -1.123538437260016e-33])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
