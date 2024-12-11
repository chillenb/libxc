
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_gga_c_zvpbeloc_H_2_zk():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_c_zvpbeloc", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([-2.253234482960566e-03, -9.099451119757799e-04, -2.578035558424197e-05, -2.977336639926227e-14, -6.034041314109771e-189])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_gga_c_zvpbeloc_H_2_vrho():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_c_zvpbeloc", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([-4.568081472507381e-03, -2.892876480488656e-02, -3.554124567192536e-03, 4.780279331391025e-01, -2.097881899454414e-04, 2.058503937272175e-02, -4.990451061683640e-13, -2.596098041756613e-12, -8.628491235370705e-187, -1.068441823342558e-185])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_gga_c_zvpbeloc_H_2_vsigma():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_c_zvpbeloc", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = numpy.asarray([6.729905055417379e-04, 1.345981011083476e-03, 6.729905055417379e-04, 7.843038023923899e-04, 1.568607604784780e-03, 7.843038023923899e-04, 3.182303664800130e-04, 6.364607329600260e-04, 3.182303664800130e-04, 3.359527329565913e-11, 6.719054659131826e-11, 3.359527329565913e-11, 5.138480516988623e-182, 1.027696103397725e-181, 5.138480516988623e-182])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
