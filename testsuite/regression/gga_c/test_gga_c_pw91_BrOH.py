
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_gga_c_pw91_BrOH_1_zk():
    # Prepare the input
    inp = test_data["BrOH"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_c_pw91", 1)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-6.798707282725298e-02, -6.798770942036679e-02, -6.798966139275607e-02, -6.797925541055985e-02, -6.798741596694202e-02, -6.798741596694202e-02, -4.962674717056072e-02, -4.963038431519452e-02, -4.973070487590813e-02, -4.955921966971228e-02, -4.962864237749277e-02, -4.962864237749277e-02, -2.900592955322842e-02, -2.879826338968105e-02, -2.336563980646309e-02, -2.366671068768745e-02, -2.893105733951320e-02, -2.893105733951320e-02, -9.894811837291882e-03, -1.060755836081151e-02, -3.595506846717085e-02, -2.750711273722335e-03, -1.011456940390596e-02, -1.011456940390596e-02, -1.274671874835970e-07, -1.577528308451522e-07, -3.439857655083025e-05, -3.099574129581132e-09, -1.562921282004650e-07, -1.562921282004650e-07, -6.637692393820756e-02, -6.654953293253377e-02, -6.639414343310218e-02, -6.652835536219383e-02, -6.646588812067698e-02, -6.646588812067698e-02, -2.759905302559351e-02, -2.807877332368053e-02, -2.695785426567699e-02, -2.732983694320268e-02, -2.877565297376693e-02, -2.877565297376693e-02, -4.053746580199043e-02, -5.664861344226255e-02, -3.910621171628386e-02, -5.151777781832984e-02, -4.287163559781561e-02, -4.287163559781561e-02, -7.094086747390803e-04, -4.977974101849613e-03, -7.792056362124501e-04, -7.331287122737190e-02, -1.559392544719730e-03, -1.559392544719730e-03, -2.783115089112498e-09, -5.177050198007926e-09, -3.886775611084420e-09, -1.495330776796902e-04, -4.971379402135864e-09, -4.971379402135864e-09, -6.148543944384161e-02, -5.659386859475513e-02, -5.825603064667648e-02, -5.961109124203781e-02, -5.892593454251035e-02, -5.892593454251035e-02, -6.164430770734226e-02, -3.106884254314617e-02, -3.784799982678925e-02, -4.545695181176174e-02, -4.148319828193563e-02, -4.148319828193563e-02, -5.672651414985271e-02, -8.481701095927661e-03, -1.364405758111860e-02, -2.602432420889634e-02, -1.941526239322075e-02, -1.941526239322075e-02, -2.929862333941243e-02, -2.057274793599427e-05, -7.913095206612068e-05, -3.043150099982610e-02, -4.849368105910046e-04, -4.849368105910102e-04, -1.963852436691251e-07, -2.014496731118286e-11, -3.630627446660553e-10, -4.889508485093251e-04, -3.409673078337473e-09, -3.409673077868432e-09]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_gga_c_pw91_BrOH_1_vrho():
    # Prepare the input
    inp = test_data["BrOH"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_c_pw91", 1)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-1.341988079736190e-01, -1.341994429652353e-01, -1.342013981382557e-01, -1.341910180293061e-01, -1.341991501823980e-01, -1.341991501823980e-01, -1.103525184772835e-01, -1.103559585047839e-01, -1.104509813488942e-01, -1.102896433809216e-01, -1.103543353348284e-01, -1.103543353348284e-01, -7.707685616095580e-02, -7.685184042311899e-02, -6.998118798110635e-02, -7.046145026288350e-02, -7.699598002060364e-02, -7.699598002060364e-02, -3.784249635116025e-02, -3.955346429637522e-02, -8.525928257277508e-02, -1.400453203553104e-02, -3.837962256252925e-02, -3.837962256252925e-02, -8.281096661579069e-07, -1.024295394623794e-06, -2.161990148016436e-04, -2.027807133227877e-08, -1.014896787682242e-06, -1.014896787682242e-06, -1.273951234820364e-01, -1.275337088879384e-01, -1.274089729060973e-01, -1.275167331245813e-01, -1.274665884309421e-01, -1.274665884309421e-01, -8.193098753401790e-02, -8.268033459516676e-02, -8.093362661654203e-02, -8.152793588623056e-02, -8.371742264432940e-02, -8.371742264432940e-02, -8.227342602157003e-02, -8.197977685399790e-02, -8.080462017936792e-02, -8.114905268915094e-02, -8.357102212329601e-02, -8.357102212329601e-02, -4.087904734634776e-03, -2.318346181011108e-02, -4.462477079558460e-03, -1.151804258133944e-01, -8.464808069617509e-03, -8.464808069617509e-03, -1.821245011933601e-08, -3.384326128265618e-08, -2.545815456478724e-08, -9.149237373879956e-04, -3.251926769369937e-08, -3.251926768791276e-08, -7.533191494262563e-02, -7.968767698632244e-02, -7.845302147555876e-02, -7.725402204248516e-02, -7.788287378519330e-02, -7.788287378519330e-02, -7.348501422510478e-02, -7.541591622176860e-02, -7.947633491139341e-02, -8.113364648496096e-02, -8.063168576583302e-02, -8.063168576583302e-02, -8.379841183638771e-02, -3.499891700304263e-02, -4.811170238395431e-02, -6.659621089640638e-02, -5.845273298087786e-02, -5.845273298087786e-02, -7.268218400173536e-02, -1.300521706435532e-04, -4.909471019676850e-04, -6.797523840347079e-02, -2.846428254983709e-03, -2.846428254983722e-03, -1.273793722206383e-06, -1.327606591655663e-10, -2.384191212379323e-09, -2.867109354386857e-03, -2.232925866065417e-08, -2.232925865715083e-08]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_gga_c_pw91_BrOH_1_vsigma():
    # Prepare the input
    inp = test_data["BrOH"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_c_pw91", 1)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [1.624713962750295e-10, 1.624736057666580e-10, 1.624759845702284e-10, 1.624399606873697e-10, 1.624726239880307e-10, 1.624726239880307e-10, 9.411297547708813e-07, 9.412696546930496e-07, 9.449355609906323e-07, 9.376567734717878e-07, 9.411820306769598e-07, 9.411820306769598e-07, 1.581868562976307e-03, 1.571542084689895e-03, 1.291711984034332e-03, 1.266589635439464e-03, 1.578181730109384e-03, 1.578181730109384e-03, 2.123016781761526e-01, 2.195247067183144e-01, 9.143240755882935e-04, 1.666547053455773e-01, 2.147645038067647e-01, 2.147645038067647e-01, 3.027703933567744e-02, 3.265981668679223e-02, 7.107080874623645e-02, 5.597560588352466e-03, 3.403228820830883e-02, 3.403228820830883e-02, 2.378018971086042e-07, 2.392845629811080e-07, 2.379481564469327e-07, 2.391009258166926e-07, 2.385665565166371e-07, 2.385665565166371e-07, 6.127010209747654e-06, 6.080342093372883e-06, 5.982600938406592e-06, 5.947982905419917e-06, 6.286932704929426e-06, 6.286932704929426e-06, 5.661696118412476e-03, 7.489748778049365e-03, 6.888495691722947e-03, 9.621073950968031e-03, 5.211516022725351e-03, 5.211516022725351e-03, 1.065401448412247e-01, 7.569978174770362e-02, 1.308446587885300e-01, 5.394689458573978e-05, 1.644465839036006e-01, 1.644465839036006e-01, 5.766128434631843e-03, 7.194625004830286e-03, 2.170296604116247e-02, 1.168186255560364e-01, 1.110002757305028e-02, 1.110002757297422e-02, 1.153325148701933e-02, 9.904908924220262e-03, 1.047536357511254e-02, 1.092865902488881e-02, 1.070255755706367e-02, 1.070255755706367e-02, 1.368807171834207e-02, 7.296588602948469e-03, 8.227474596608999e-03, 9.561138215236688e-03, 8.839652990259130e-03, 8.839652990259132e-03, 5.934753521829750e-03, 5.184247692971749e-02, 4.354791727687075e-02, 3.323263802626322e-02, 3.943474033382435e-02, 3.943474033382439e-02, 1.111219715690525e-02, 4.880686079454798e-02, 7.750116046365915e-02, 5.633050631792698e-02, 1.747046596044395e-01, 1.747046596044408e-01, 2.513229226678030e-02, 1.787926616491866e-03, 4.362488894437905e-03, 2.248505653047226e-01, 1.498651348709771e-02, 1.498651348315700e-02]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05