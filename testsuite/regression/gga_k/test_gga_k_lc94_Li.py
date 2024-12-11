
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_gga_k_lc94_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_k_lc94", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([1.644944951546692e+01, 8.176591948450980e+00, 6.811464940884575e-01, 1.325079583243506e-01, 2.695992126523974e-02, 1.117341036547990e-03, 3.457774630846251e-10])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_gga_k_lc94_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_k_lc94", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([2.596560130198935e+01, 2.601305222640973e+01, 1.237253687711578e+01, 1.239389107061039e+01, 6.954676496283656e-01, 6.949790359726919e-01, 2.136979187386189e-01, 3.838615710717749e-03, 3.180413052501541e-02, 1.083379754808156e-08, 4.621797728063265e-03, 4.385824140431311e-03, 1.836616889053116e-09, 5.608737678616597e-10])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_gga_k_lc94_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_k_lc94", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = numpy.asarray([2.444001670479915e-03, 0.000000000000000e+00, 2.437983771952554e-03, 6.553603460532175e-03, 0.000000000000000e+00, 6.538283954261730e-03, 2.132981847567082e-01, 0.000000000000000e+00, 2.132140323807903e-01, 3.513915192023671e+00, 0.000000000000000e+00, -2.176845481440139e+01, 2.872948211758100e+01, 0.000000000000000e+00, -5.073753355402905e+00, -2.253613448995860e+01, 0.000000000000000e+00, -2.120510000072587e+01, -2.462067302604689e+00, 0.000000000000000e+00, -2.503489377892144e+00])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
