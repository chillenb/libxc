
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_lda_c_2d_amgb_BrOH_cation_restr_1_zk():
    # Prepare the input
    inp = test_data["BrOH_cation_restr"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("lda_c_2d_amgb", 1)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-1.905654107674321e-01, -1.905654174179582e-01, -1.905654490366279e-01, -1.905653500461962e-01, -1.905654016244644e-01, -1.905654016244644e-01, -1.768064374477722e-01, -1.768064730553962e-01, -1.768077002273191e-01, -1.768086989864427e-01, -1.768068778115274e-01, -1.768068778115274e-01, -1.220574142366990e-01, -1.219994351365025e-01, -1.205765547357921e-01, -1.210151396415753e-01, -1.209222930082139e-01, -1.209222930082139e-01, -4.608809690310589e-02, -4.679054430652110e-02, -1.298645538546976e-01, -3.445804592287061e-02, -3.906019753538134e-02, -3.906019753538131e-02, -5.359920648397853e-04, -5.780824671941520e-04, -6.656442274338314e-03, -2.375230131823336e-04, -3.334534851161530e-04, -3.334534851161530e-04, -1.823439436910625e-01, -1.823460015994923e-01, -1.823440462037191e-01, -1.823458628451125e-01, -1.823449827571728e-01, -1.823449827571728e-01, -1.644838318302772e-01, -1.646781195250251e-01, -1.643403960328795e-01, -1.645138550003420e-01, -1.646599133559080e-01, -1.646599133559080e-01, -1.148400266911141e-01, -1.193495529116400e-01, -1.104978211779277e-01, -1.126782567655615e-01, -1.157211612241478e-01, -1.157211612241478e-01, -2.267589069242767e-02, -4.615870209027598e-02, -2.065589144826757e-02, -1.639593539602085e-01, -2.735502728893605e-02, -2.735502728893605e-02, -1.614661479534540e-04, -2.296496866467657e-04, -1.540122338958838e-04, -1.260446493385650e-02, -2.032384444195234e-04, -2.032384444195234e-04, -1.151073857605880e-01, -1.147898024426715e-01, -1.149018336300494e-01, -1.149938823317582e-01, -1.149478334897123e-01, -1.149478334897123e-01, -1.135848305310587e-01, -1.048996019280084e-01, -1.075585207392178e-01, -1.100517749892693e-01, -1.087956546765428e-01, -1.087956546765428e-01, -1.218591015684453e-01, -5.782863915145836e-02, -6.791310865997473e-02, -8.423535916366243e-02, -7.594189755919217e-02, -7.594189755919227e-02, -1.000634104252658e-01, -6.272012911927000e-03, -9.498342002354443e-03, -8.204728699036473e-02, -1.708975529908982e-02, -1.708975529908988e-02, -8.900522457985227e-04, -3.186278875108584e-05, -9.705541661081396e-05, -1.580386957286031e-02, -1.810641040242145e-04, -1.810641040234096e-04]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_lda_c_2d_amgb_BrOH_cation_restr_1_vrho():
    # Prepare the input
    inp = test_data["BrOH_cation_restr"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("lda_c_2d_amgb", 1)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-1.913603019073399e-01, -1.913603059446782e-01, -1.913603251394327e-01, -1.913602650452413e-01, -1.913602963569128e-01, -1.913602963569128e-01, -1.823055226803806e-01, -1.823055473978124e-01, -1.823063992497548e-01, -1.823070925423706e-01, -1.823058283634527e-01, -1.823058283634527e-01, -1.395729060956019e-01, -1.395228665954116e-01, -1.382912453679433e-01, -1.386716150902573e-01, -1.385911474696982e-01, -1.385911474696982e-01, -6.158976687238316e-02, -6.244331239585535e-02, -1.462097967927647e-01, -4.707800003709958e-02, -5.290544280064962e-02, -5.290544280064956e-02, -7.998635538217154e-04, -8.624258998570300e-04, -9.662215765462483e-03, -3.552546736006535e-04, -4.983437881945399e-04, -4.983437881945399e-04, -1.860811090327924e-01, -1.860824847237044e-01, -1.860811775619882e-01, -1.860823919685058e-01, -1.860818036407185e-01, -1.860818036407185e-01, -1.734701562318749e-01, -1.736134283043428e-01, -1.733643084077438e-01, -1.734923036359552e-01, -1.736000076462858e-01, -1.736000076462858e-01, -1.332539773548578e-01, -1.372235806166306e-01, -1.293612649218623e-01, -1.313248282624288e-01, -1.340353600029356e-01, -1.340353600029356e-01, -3.167445813506012e-02, -6.167567886373140e-02, -2.896692610140898e-02, -1.730828114883762e-01, -3.787271566852348e-02, -3.787271566852348e-02, -2.416451716842826e-04, -3.435026658143315e-04, -2.305265101188557e-04, -1.798734048422592e-02, -3.040685209553148e-04, -3.040685209553148e-04, -1.334913685828627e-01, -1.332093534544163e-01, -1.333088797082650e-01, -1.333906195204694e-01, -1.333497317431639e-01, -1.333497317431639e-01, -1.321359595648813e-01, -1.242353128790499e-01, -1.266853641308749e-01, -1.289573492717853e-01, -1.278157404188814e-01, -1.278157404188814e-01, -1.394017029895291e-01, -7.550774604903998e-02, -8.687461918815213e-02, -1.041611070448159e-01, -9.554506341489390e-02, -9.554506341489390e-02, -1.197049226716722e-01, -9.116254016648465e-03, -1.366624827166351e-02, -1.019206755789709e-01, -2.414115777349953e-02, -2.414115777349954e-02, -1.325172920872276e-03, -4.811272682113985e-05, -1.453662200612724e-04, -2.238668823887218e-02, -2.709359479712153e-04, -2.709359479717115e-04]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05