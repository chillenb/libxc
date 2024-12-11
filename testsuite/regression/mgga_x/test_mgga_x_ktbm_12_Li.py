
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_mgga_x_ktbm_12_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_ktbm_12", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([-1.958354757207856e+00, -1.292080731729006e+00, -2.681909854813155e-01, -1.806722881680783e-01, -5.655976306003765e-02, -1.157777898356212e-02, -2.148828636442340e-04])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_mgga_x_ktbm_12_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_ktbm_12", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([-2.781415301726221e+00, -2.784109337377098e+00, -1.925327128500074e+00, -1.927071031036508e+00, -3.404627771788263e-01, -3.400598908992053e-01, -2.508782539726818e-01, -1.395175139242143e-02, -7.541764660430798e-02, -4.424650666790693e-04, -1.467056090072649e-02, -1.456455132054160e-02, -2.955412128623213e-04, -2.153116354845804e-04])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_x_ktbm_12_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_ktbm_12", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = numpy.asarray([-6.332056053475334e-04, 0.000000000000000e+00, -6.310616852965168e-04, -2.362597571400718e-03, 0.000000000000000e+00, -2.356321655721757e-03, -4.061466238920103e-02, 0.000000000000000e+00, -4.257407154507757e-02, -9.938646882801287e+00, 0.000000000000000e+00, -2.006264816296378e+01, -6.852995469558834e+01, 0.000000000000000e+00, -5.025023404344246e+04, -3.730875290555162e-01, 0.000000000000000e+00, -1.793836565628586e+01, -7.609321334171347e-01, 0.000000000000000e+00, 4.647219864106238e+04])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_x_ktbm_12_Li_2_vtau():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_ktbm_12", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vtau"].flatten()
    ref_tgt = numpy.asarray([2.445810129954252e-02, 2.443713872600277e-02, 2.974840552118589e-02, 2.975330882747791e-02, -3.155527930916498e-03, -3.219892763152374e-03, 2.818376774127196e-01, 2.565295809146883e-04, 3.865316113552119e-05, 2.047378506339215e-05, 5.540933518123740e-06, 2.609542124052305e-04, 9.238937726060493e-11, -1.013866200074583e-11])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
