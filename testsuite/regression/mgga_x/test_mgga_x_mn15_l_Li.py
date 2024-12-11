
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_mgga_x_mn15_l_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_mn15_l", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([-1.300412051187539e+00, -1.219640473101470e+00, 4.343669581853517e-02, -1.638311684141456e-01, -2.896246239212740e-02, -2.309148357448394e-02, -3.386121602657094e-04])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_mgga_x_mn15_l_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_mn15_l", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([-3.948537704181185e-01, -3.901988917829586e-01, -1.400313951110305e+00, -1.398645289558582e+00, -2.784756590885476e-01, -2.875676610622709e-01, -2.911773546772938e-01, -2.861005258946357e-02, -1.123365874451646e-01, -1.047497795020582e-03, -2.996068615641109e-02, -2.967065531763801e-02, -7.007403445819201e-04, 2.434987030011256e-04])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_x_mn15_l_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_mn15_l", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = numpy.asarray([-1.141284835724559e-03, 0.000000000000000e+00, -1.137780814231160e-03, -2.599453748732595e-03, 0.000000000000000e+00, -2.597473265130826e-03, -9.351831367183647e-01, 0.000000000000000e+00, -9.344639181183055e-01, 3.971946981899085e+01, 0.000000000000000e+00, -1.060669746726801e+00, -5.783887704225166e+02, 0.000000000000000e+00, -7.304543375520956e+00, -4.521741204079812e-04, 0.000000000000000e+00, -1.003139506221568e+00, -3.336774952254383e-10, 0.000000000000000e+00, -1.722907875553752e+13])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_x_mn15_l_Li_2_vtau():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_mn15_l", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vtau"].flatten()
    ref_tgt = numpy.asarray([-1.585336906515587e-01, -1.590359079811931e-01, -2.561159300679617e-02, -2.595205743134399e-02, 6.082329574514136e-02, 6.289264508287534e-02, 1.509320766990459e+00, 7.314209482050236e-06, 6.247459526036514e-01, 1.450354763207829e-09, 3.606511271792671e-09, 7.905968280815082e-06, 1.971959110442651e-20, 1.896304964575415e-10])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
