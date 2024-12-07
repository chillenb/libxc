
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_gga_c_cs1_H_2_zk():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_c_cs1", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-1.241228154697412e-02, -1.159742005356726e-02, -7.392546973375932e-03, 1.378336127298605e-02, 1.698217285720566e-03]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_gga_c_cs1_H_2_vrho():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_c_cs1", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-1.383271813855897e-02, -1.044959846891935e-01, -1.474030809300259e-02, -1.004993074400815e-01, -1.787783343049169e-02, -7.958278681447356e-02, -7.625063667499026e-03, -3.271800579059599e-02, 2.251138526864826e-03, -1.884876138847289e-03]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_gga_c_cs1_H_2_vsigma():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_c_cs1", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [1.010697016780541e-05, -1.972772559539766e-23, 4.832690501635346e+25, 7.987841355044313e-04, -1.519613117609735e-21, 3.483793204228920e+25, 2.021354641774754e-02, 3.219899787038242e-20, 6.970892089197868e+24, 2.817551173751784e+00, 2.841166783170105e-18, 1.327706815048277e+23, 4.867179795587189e+00, 5.828809636571210e-14, 1.460398849255396e+22]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
