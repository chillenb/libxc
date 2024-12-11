
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_mgga_x_mbr_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_mbr", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-1.667182588896387e+00, -1.163528414249882e+00, -3.285872337863665e-01, -1.504018256889672e-01, -6.437667063884213e-02, -3.249718287365569e-01, -6.069597109204938e+00]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_mgga_x_mbr_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_mbr", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-2.207526094049141e+00, -2.209731915346792e+00, -1.530137664564037e+00, -1.531454575768338e+00, -3.548695544007818e-01, -3.540457793015103e-01, -1.997872784047159e-01, -4.821059165191918e-02, -7.613440482789377e-02, -1.263142957242692e-02, -1.815306577119726e-01, -4.944379318260444e-02, -1.913554548068817e+00, -1.305242410402354e-02]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_x_mbr_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_mbr", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [-4.823298088321271e-04, 0.000000000000000e+00, -4.811352469657244e-04, -2.168636708034858e-03, 0.000000000000000e+00, -2.160819868886726e-03, -7.943235040119548e-01, 0.000000000000000e+00, -8.060179436419146e-01, -7.099617990899123e+00, 0.000000000000000e+00, -7.708923692798138e+02, -4.248924342678481e+02, 0.000000000000000e+00, -2.522969449269815e+07, -7.174797372104202e+01, 0.000000000000000e+00, -6.721928604484855e+02, -1.250170270408591e+05, 0.000000000000000e+00, -2.210226236928907e+09]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_x_mbr_Li_2_vtau():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_mbr", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vtau"].flatten()
    ref_tgt = [-2.004610429840794e-03, -2.001121587877980e-03, -3.010561641177324e-03, -3.007376153730942e-03, -1.528801532792490e-02, -1.549330993680480e-02, -2.178612519694554e-02, -3.940679862211121e-03, -8.122752233659597e-02, -4.140719680214862e-03, -4.290012518083431e-04, -3.908127788757420e-03, -6.115469122747638e-06, -7.707479439865639e-03]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
