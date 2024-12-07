
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_hyb_gga_xc_whpbe0_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_gga_xc_whpbe0", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-4.806122003462828e-01, -3.364062433219691e-01, -7.962600424507066e-02, -3.361759650517075e-02, -4.399239061647299e-03, -1.927364327235272e-05, -1.237669350381907e-10]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_hyb_gga_xc_whpbe0_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_gga_xc_whpbe0", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-6.490180785118411e-01, -6.494254101334942e-01, -4.536962712632326e-01, -4.539435107388503e-01, -8.873747561767650e-02, -8.878528554137502e-02, -5.288443402582700e-02, -9.767474852576256e-02, -1.020511389599938e-02, 3.428185822405095e-01, -4.043850608851033e-05, -3.947364987120443e-05, -2.980017138447229e-10, -1.070702879680258e-10]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_hyb_gga_xc_whpbe0_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_gga_xc_whpbe0", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [-1.704102993506767e-05, 9.190971700708733e-05, -1.682390212670922e-05, -1.004078880281227e-04, 2.980993506782570e-04, -9.960967827425660e-05, -1.526717037885362e-02, 6.249948659585063e-03, -1.522256549898600e-02, 2.952200645455288e+00, 6.762268918356340e+00, 3.381029382522827e+00, 6.249034463277400e+00, 2.258698854598489e+01, 1.129349427295099e+01, 2.859296211789336e-05, 3.357174600576258e-04, 4.268622265764520e-05, 1.606537652049720e-06, 3.212885779437900e-06, 1.606541427663688e-06]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
