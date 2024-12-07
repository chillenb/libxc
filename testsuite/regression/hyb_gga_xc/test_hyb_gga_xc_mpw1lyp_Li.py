
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_hyb_gga_xc_mpw1lyp_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_gga_xc_mpw1lyp", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-1.407759916267564e+00, -1.018146698009047e+00, -3.060996230317823e-01, -1.204552241048576e-01, -6.076812033921984e-02, -3.362245292764437e-03, -3.008690882436738e-05]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_hyb_gga_xc_mpw1lyp_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_gga_xc_mpw1lyp", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-1.745456805662656e+00, -1.746878152032170e+00, -1.208529190903280e+00, -1.209379764034749e+00, -3.736505482287809e-01, -3.740633095834032e-01, -1.537978619672893e-01, -8.641660050991939e-02, -5.596151171288118e-02, -3.030018233145571e-02, -7.531399205308004e-03, -7.385678308072158e-03, -2.100523858905803e-05, -9.370439347952903e-05]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_hyb_gga_xc_mpw1lyp_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_gga_xc_mpw1lyp", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [-2.004462207917374e-04, 5.222815421851711e-06, -1.998874136765783e-04, -7.344784422653205e-04, 3.646941789248587e-05, -7.327143040864123e-04, -4.184816859229185e-02, 4.773762863586187e-02, -4.162360060245511e-02, -3.341909283122727e+00, 4.596134769453040e+00, 2.910680288004614e+01, -5.484293794077895e+01, 2.356939734329661e+01, 3.634198707909246e+02, 2.576724144182902e+01, 7.936097321777658e-02, 2.421145231474783e+01, 2.871433983922068e+02, 0.000000000000000e+00, 4.404424295670549e+02]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
