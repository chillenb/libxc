
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_gga_x_rpw86_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_rpw86", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-1.801221257335163e+00, -1.307665304356224e+00, -4.280268155925332e-01, -1.598578440519332e-01, -8.357275336745386e-02, -4.918351910041442e-02, -3.814340998249643e-03]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_gga_x_rpw86_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_rpw86", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-2.183167576360579e+00, -2.185379572352732e+00, -1.455362332990779e+00, -1.456671100556150e+00, -4.177420944011712e-01, -4.177577643184586e-01, -2.030529291670760e-01, -3.848084703938009e-02, -8.249308437904534e-02, -3.991677138750301e-03, -3.963876229669393e-02, -3.971673719835214e-02, -3.234305074550403e-03, -2.542679096333602e-03]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_gga_x_rpw86_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_rpw86", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [-3.726688995319385e-04, 0.000000000000000e+00, -3.712490837828276e-04, -1.514817393109984e-03, 0.000000000000000e+00, -1.510188733301009e-03, -7.418327899600048e-02, 0.000000000000000e+00, -7.407366019520845e-02, -4.934233217736829e+00, 0.000000000000000e+00, -2.416631494612772e+02, -6.332148683651166e+01, 0.000000000000000e+00, -2.023706543604857e+06, -2.153753567532398e+02, 0.000000000000000e+00, -2.133509888775784e+02, -4.694867223787501e+06, 0.000000000000000e+00, -1.229067417938645e+07]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
