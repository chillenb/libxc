
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_gga_xc_mohlyp_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_xc_mohlyp", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-1.818124301282109e+00, -1.290656683895239e+00, -4.590276508067057e-01, -1.661762452017006e-01, -8.403866120595666e-02, -3.162149626084685e-02, -6.047033425809060e-04]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_gga_xc_mohlyp_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_xc_mohlyp", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-2.370326343659652e+00, -2.372311114583620e+00, -1.621838500189281e+00, -1.623120641961399e+00, -3.567856739212019e-01, -3.573133628379753e-01, -2.188686224134406e-01, -1.401082064086289e-01, -6.205683211106285e-02, -5.344031068097252e-02, -4.190550332635939e-02, -4.174058032296498e-02, -8.397996315522495e-04, -7.100229798987490e-04]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_gga_xc_mohlyp_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_xc_mohlyp", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [-5.965745851345558e-05, 2.611407710925856e-06, -5.939765449468962e-05, -4.437639261140195e-04, 1.823470894624294e-05, -4.421432046112739e-04, -1.340220985998879e-01, 2.386881431793093e-02, -1.337109055454797e-01, -5.873976035888951e-01, 2.298067384726520e+00, 1.006248884857863e+00, -1.077350155222652e+02, 1.178469867164830e+01, 4.244173278803880e+00, -7.092058419813996e-01, 3.968048660888829e-02, -6.608270421454325e-01, -3.344520265427838e+00, 0.000000000000000e+00, -4.787337231563085e+00]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
