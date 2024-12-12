
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_mgga_c_mn15_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_mn15", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([-1.607716086290737e-01, -8.471418769751494e-02, 6.837382934087993e-02, -4.312154479343108e-02, 1.289448759925678e-02, -1.922901299375069e-03, -4.775266028068120e-05])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_mgga_c_mn15_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_mn15", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([-4.039142010341299e-01, -4.035588787436167e-01, -2.921465244755226e-01, -2.919799028605005e-01, 5.813521120835001e-02, 5.817455827319325e-02, -1.160567024251476e-01, -3.174192749366093e-01, -4.627616708033183e-03, 2.329721052491062e+00, -2.414497837420723e-03, -2.441621474998957e-03, -5.617609056740022e-05, -8.243054259892993e-05])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_c_mn15_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_mn15", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = numpy.asarray([1.610729824438502e-04, 3.221459648877004e-04, 1.610729824438502e-04, 1.740988003829623e-04, 3.481976007659246e-04, 1.740988003829623e-04, 2.184440479978687e-02, 4.368880959957375e-02, 2.184440479978687e-02, 1.152673220749618e+01, 2.305346441499236e+01, 1.152673220749618e+01, 7.175416541586684e+01, 1.435083308317336e+02, 7.175416541586684e+01, 1.780262600204291e-03, 3.560525200488931e-03, 1.780262600204291e-03, 1.704113076143791e-05, 3.408231551877447e-05, 1.704113076143791e-05])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_c_mn15_Li_2_vtau():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_mn15", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vtau"].flatten()
    ref_tgt = numpy.asarray([7.343180172615965e-03, 7.343180172615953e-03, 1.990575869951162e-02, 1.990575869951162e-02, -1.228748409479562e-02, -1.228748409479562e-02, 1.179609549014607e+00, 1.179609549014350e+00, -9.654082833853507e-02, -9.654082827198847e-02, -2.569197986312393e-08, -2.569197986312394e-08, -6.789917492756186e-20, -6.790563720746361e-20])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
