
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_gga_c_xpbe_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_c_xpbe", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-6.060898885461793e-02, -4.527799266858566e-02, -3.418514298632165e-03, -1.446947192500918e-02, -1.028163975438139e-03, -4.482135393737985e-09, -9.958396954279358e-17]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_gga_c_xpbe_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_c_xpbe", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-1.077369689587807e-01, -1.076147088477595e-01, -9.374956647488052e-02, -9.365687096589292e-02, -1.729305817499749e-02, -1.729922891084691e-02, -2.442315502109415e-02, -8.654846409476122e-02, -5.225417176725308e-03, 3.574565611004774e-01, -2.909657846076098e-08, -2.924214662592174e-08, -6.283832503818643e-16, -7.437694665886341e-16]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_gga_c_xpbe_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_c_xpbe", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [3.717195094674096e-05, 7.434390189348191e-05, 3.717195094674096e-05, 1.195768260490649e-04, 2.391536520981298e-04, 1.195768260490649e-04, 3.276037905114665e-03, 6.552075810229331e-03, 3.276037905114665e-03, 3.875730595384028e+00, 7.751461190768056e+00, 3.875730595384028e+00, 8.973488593422450e+00, 1.794697718684489e+01, 8.973488593422450e+00, 9.888100481730770e-05, 1.977620096104275e-04, 9.888100481730770e-05, 8.883574924146648e-07, 1.776560070140016e-06, 8.883574924146648e-07]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
