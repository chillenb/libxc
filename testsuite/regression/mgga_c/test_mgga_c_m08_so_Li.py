
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_mgga_c_m08_so_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_m08_so", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-2.139024636468379e-01, -1.181134693864477e-01, -1.995992848245685e-01, -4.182667836673438e-02, -3.796055151246079e-02, -3.948869978531433e-02, -9.799458568975964e-04]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_mgga_c_m08_so_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_m08_so", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-1.199265685512045e-02, -1.162210794066178e-02, -4.437417726617599e-01, -4.435460695211186e-01, -1.893848069967969e-02, -1.911366295803554e-02, -9.675734504888272e-03, -2.485583790899780e-01, -1.781976924041577e-02, -2.201862461500710e-01, -4.962545399189820e-02, -5.018243448765679e-02, -1.152805453892758e-03, -1.691580494719088e-03]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_c_m08_so_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_m08_so", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [6.402467349239047e-04, 1.280493469847809e-03, 6.402467349239047e-04, 7.482557668320964e-04, 1.496511533664193e-03, 7.482557668320964e-04, 7.860645329158927e-01, 1.572129065831785e+00, 7.860645329158927e-01, 1.948084148532276e+01, 3.896168297064552e+01, 1.948084148532276e+01, 7.364276424329736e+02, 1.472855284865947e+03, 7.364276424329736e+02, -4.475379192272974e-08, -8.950758626838992e-08, -4.475379192272974e-08, -4.718887703851019e-16, 8.347242043006633e-15, -4.718887703851019e-16]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_c_m08_so_Li_2_vtau():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_m08_so", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vtau"].flatten()
    ref_tgt = [-3.173316397585907e-02, -3.173316397585900e-02, 4.580072822454370e-02, 4.580072822454370e-02, -3.845037815996159e-02, -3.845037815996159e-02, -1.056870093623146e+00, -1.056870093622921e+00, -2.308661976871572e-01, -2.308661975280074e-01, -1.356981622033984e-07, -1.356981620874713e-07, -3.569246435322067e-19, -3.585789871870490e-19]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
