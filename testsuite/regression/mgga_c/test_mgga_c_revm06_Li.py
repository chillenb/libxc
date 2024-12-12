
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_mgga_c_revm06_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_revm06", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([-6.784900100154619e-02, -5.581923947848879e-02, 7.314382810123919e-03, 2.223932093147485e-04, 7.338468095438852e-08, 2.344645596802505e-02, 5.639361640164805e-04])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_mgga_c_revm06_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_revm06", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([-8.165564406003734e-02, -8.135505232775267e-02, -6.277209345565198e-02, -6.243666156579044e-02, -8.002742525912178e-02, -7.889698311482214e-02, -4.049825611237210e-04, 6.914299636342843e-01, 1.125159554109006e-02, 4.174968765424958e-01, 3.487517832694113e-02, 3.335880777426976e-02, 6.422490371617828e-04, 1.256246966524829e-03])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_c_revm06_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_revm06", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = numpy.asarray([-1.787979010461381e-05, 0.000000000000000e+00, -1.778959373282284e-05, -1.802595010840745e-04, 0.000000000000000e+00, -1.802740003130841e-04, 1.905835811085153e-02, 0.000000000000000e+00, 1.645918843322740e-02, 5.409348988640081e-01, 0.000000000000000e+00, -7.970083990831728e+01, -6.565618947870561e+01, 0.000000000000000e+00, -8.753107386661471e+05, -2.224994163101343e+00, 0.000000000000000e+00, -1.930306394230921e+02, -1.051642717310018e+01, 0.000000000000000e+00, -4.194897595003526e+06])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_c_revm06_Li_2_vtau():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_revm06", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vtau"].flatten()
    ref_tgt = numpy.asarray([2.508670543603678e-03, 2.498750775542548e-03, 5.166165533263339e-03, 5.163706835467791e-03, 9.008081578857902e-03, 9.828090974865580e-03, -1.912259339587041e-02, 2.753653556492756e-03, 1.570151910786815e-01, 3.631734307709729e-04, 1.349998746453379e-06, 2.831382738507207e-03, 1.327958654631000e-14, 1.829966171234195e-04])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
