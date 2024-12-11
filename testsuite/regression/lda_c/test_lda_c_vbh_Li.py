
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_lda_c_vbh_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("lda_c_vbh", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([-1.112110095547117e-01, -1.021593141750027e-01, -6.763308871899304e-02, -3.498161022585779e-02, -2.443336863975705e-02, -1.108383210135501e-02, -2.614647361183199e-04])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_lda_c_vbh_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("lda_c_vbh", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([-1.196029477332483e-01, -1.194013936795466e-01, -1.104890901823217e-01, -1.103180978620517e-01, -7.540271533252356e-02, -7.545175726094916e-02, -3.890856342656562e-02, -1.214447289344952e-01, -2.802247468252016e-02, -6.718519878729629e-02, -1.404381228127695e-02, -1.404883568576870e-02, -3.758092223459343e-04, -2.706221752240713e-04])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
