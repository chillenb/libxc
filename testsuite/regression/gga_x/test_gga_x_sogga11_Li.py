
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_gga_x_sogga11_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_sogga11", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-1.775129343678530e+00, -1.258541551826166e+00, -3.705050875093007e-01, -1.589454777465956e-01, -6.692162961297134e-02, -1.682844988385264e-01, -3.191586608168198e-03]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_gga_x_sogga11_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_sogga11", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-2.258215739562814e+00, -2.260334293496306e+00, -1.557133394861547e+00, -1.558427882865197e+00, -6.193252522896220e-01, -6.303084118345298e-01, -2.062777696390629e-01, -2.082806151860198e-01, -6.006341315570623e-02, -6.897419136902783e-03, -2.179174797289906e-01, -2.168397489280279e-01, -4.607389728345342e-03, -3.275486693260556e-03]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_gga_x_sogga11_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_sogga11", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [-1.843924445583299e-04, 0.000000000000000e+00, -1.837537119192651e-04, -6.334766189961510e-04, 0.000000000000000e+00, -6.318567823849754e-04, 6.067574215381044e-02, 0.000000000000000e+00, 6.611392706185444e-02, -2.748557351993530e+00, 0.000000000000000e+00, -5.894844035561416e+01, -6.382077386215713e+01, 0.000000000000000e+00, -3.798928569030190e+02, -5.985278437569703e+01, 0.000000000000000e+00, -5.591416345134663e+01, -2.765512467049721e+02, 0.000000000000000e+00, -3.958557881963652e+02]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
