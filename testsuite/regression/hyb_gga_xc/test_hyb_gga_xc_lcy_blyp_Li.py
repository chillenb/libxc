
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_hyb_gga_xc_lcy_blyp_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_gga_xc_lcy_blyp", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-1.529811072352407e+00, -1.018463307216642e+00, -1.190282757812775e-01, -3.092502267281993e-02, -3.018104431813423e-03, -2.098694122987986e-03, -3.003784931274436e-05]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_hyb_gga_xc_lcy_blyp_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_gga_xc_lcy_blyp", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-2.003432749068511e+00, -2.005322144551398e+00, -1.314760264994833e+00, -1.315891107195902e+00, -2.873665943540706e-01, -2.877162725550477e-01, -5.356560913180386e-02, -8.226452841564215e-02, -5.729731348393773e-03, -3.029946541911563e-02, -2.728862881776122e-03, -2.819043795330207e-03, -2.079720121395204e-05, -9.360851553213757e-05]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_hyb_gga_xc_lcy_blyp_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_gga_xc_lcy_blyp", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [-1.962651241516493e-04, 5.222815421851711e-06, -1.957717415968542e-04, -6.297960286891641e-04, 3.646941789248587e-05, -6.285896726672530e-04, 1.638115803588638e-02, 4.773762863586187e-02, 1.656887725007333e-02, -3.156369527510413e-01, 4.596134769453040e+00, 3.448481074907228e+00, -2.632307834642778e-01, 2.356939734329661e+01, 1.767704993870170e+01, 4.079229184664095e-02, 7.936097321777658e-02, 4.100717618674121e-02, -3.667739469572191e-09, 0.000000000000000e+00, -1.774316363885257e-09]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
