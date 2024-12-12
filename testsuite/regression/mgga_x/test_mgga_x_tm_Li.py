
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_mgga_x_tm_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_tm", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([-1.941167231498373e+00, -1.359991641755154e+00, -3.810491008156147e-01, -1.749370946683342e-01, -7.696889665316035e-02, -7.596817714725745e-02, -3.283654011231219e-02])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_mgga_x_tm_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_tm", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([-2.544040969894749e+00, -2.546812217909044e+00, -1.714086699477120e+00, -1.716251783316255e+00, -3.548542746595577e-01, -3.550844173567085e-01, -2.300923696611670e-01, -2.085620783475743e-02, -7.835920705307962e-02, -2.079140640245233e-03, -1.100575863700110e-01, -2.155148277148765e-02, -4.368028490904265e-02, -1.323262530378711e-03])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_x_tm_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_tm", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = numpy.asarray([-8.460572742645320e-05, 0.000000000000000e+00, -8.228431286555687e-05, -7.550097466553306e-04, 0.000000000000000e+00, -7.431093488334087e-04, -1.187167350988533e-01, 0.000000000000000e+00, -1.193030552533710e-01, -3.233022948115081e+00, 0.000000000000000e+00, -5.047770576772503e+02, -8.230979473070408e+01, 0.000000000000000e+00, -4.240807153799870e+06, 7.188759654607422e+00, 0.000000000000000e+00, -4.456805425096432e+02, 8.451602987215789e+01, 0.000000000000000e+00, -2.576093862008627e+07])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_x_tm_Li_2_vtau():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_tm", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vtau"].flatten()
    ref_tgt = numpy.asarray([9.017225279919289e-04, 7.951732406826366e-04, 6.556744784365733e-03, 6.385493248826587e-03, 1.681535670421196e-02, 1.720746352010834e-02, 9.750993654164590e-02, 3.895036744553971e-03, 1.117684887531187e-01, 1.039469549318501e-03, -1.501912709409667e-04, 3.912939835595237e-03, -4.876834669210909e-08, 6.760051167976726e-04])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
