
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_gga_x_lambda_ch_n_BrOH_1_zk():
    # Prepare the input
    inp = test_data["BrOH"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_lambda_ch_n", 1)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-2.094059962641247e+01, -2.094062370113550e+01, -2.094080722757987e+01, -2.094041147995436e+01, -2.094061168628272e+01, -2.094061168628272e+01, -3.472048202160965e+00, -3.472019253785328e+00, -3.471367080422047e+00, -3.473227414908032e+00, -3.472048103589210e+00, -3.472048103589210e+00, -6.975026864158628e-01, -6.975523237633501e-01, -7.008048028747916e-01, -7.050756424209496e-01, -6.975169884684379e-01, -6.975169884684379e-01, -2.161593748362170e-01, -2.170709518118204e-01, -8.121537840262133e-01, -1.783641568682488e-01, -2.164096745810149e-01, -2.164096745810149e-01, -1.657758270325325e-02, -1.736026785959610e-02, -6.881491416766791e-02, -7.967699738049273e-03, -1.717943972812311e-02, -1.717943972812311e-02, -5.032260928476063e+00, -5.031670991062800e+00, -5.032206888039786e+00, -5.031748068614418e+00, -5.031949306224887e+00, -5.031949306224887e+00, -2.110503738164518e+00, -2.120094460914877e+00, -2.111926948037077e+00, -2.119387684977720e+00, -2.115164915640705e+00, -2.115164915640705e+00, -5.737701897038451e-01, -5.954352407143366e-01, -5.465695795840366e-01, -5.476161549557355e-01, -5.914267907707235e-01, -5.914267907707235e-01, -1.424911185352485e-01, -2.358791783895299e-01, -1.399910873502393e-01, -1.811344855222113e+00, -1.577338689574917e-01, -1.577338689574917e-01, -7.686479450390487e-03, -8.783228419789598e-03, -6.583258151706112e-03, -9.345073623078134e-02, -8.002842494193091e-03, -8.002842494193091e-03, -5.586261939928442e-01, -5.614366171656552e-01, -5.604440344400732e-01, -5.596630130636528e-01, -5.600542479822151e-01, -5.600542479822151e-01, -5.397585155327205e-01, -5.163379837389019e-01, -5.231622318421562e-01, -5.291975447701708e-01, -5.259583400563764e-01, -5.259583400563764e-01, -6.253006415781616e-01, -2.808059315028797e-01, -3.145119113548794e-01, -3.663995034304346e-01, -3.384877688033635e-01, -3.384877688033633e-01, -4.686353054758196e-01, -6.411186953342649e-02, -8.520849506961753e-02, -3.363270838227920e-01, -1.174394173614910e-01, -1.174394173614910e-01, -1.942019562490078e-02, -2.405915945226286e-03, -4.569770654681338e-03, -1.119501750327865e-01, -6.801847929255531e-03, -6.801847929255521e-03]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_gga_x_lambda_ch_n_BrOH_1_vrho():
    # Prepare the input
    inp = test_data["BrOH"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_lambda_ch_n", 1)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-2.515935521664267e+01, -2.515944133481426e+01, -2.515983428265535e+01, -2.515842390456199e+01, -2.515940055924079e+01, -2.515940055924079e+01, -4.041575765209179e+00, -4.041613210353942e+00, -4.042809465497057e+00, -4.041605718651471e+00, -4.041612240741443e+00, -4.041612240741443e+00, -7.549861244592309e-01, -7.538807853900770e-01, -7.272668417509409e-01, -7.332157508493644e-01, -7.545838494371175e-01, -7.545838494371175e-01, -2.088472017989396e-01, -2.104922417136132e-01, -9.179098152689518e-01, -1.782282021828023e-01, -2.093103359774792e-01, -2.093103359774792e-01, -2.200133644010973e-02, -2.302958111115647e-02, -8.693885939117307e-02, -1.061407282988281e-02, -2.278992043485582e-02, -2.278992043485582e-02, -6.199521909593213e+00, -6.202167956526770e+00, -6.199792418627843e+00, -6.201849860951643e+00, -6.200878646441737e+00, -6.200878646441737e+00, -2.194057453572008e+00, -2.210541889095211e+00, -2.186547032564003e+00, -2.199288706010166e+00, -2.215374735160829e+00, -2.215374735160829e+00, -6.792200789190537e-01, -7.667546526788749e-01, -6.426926857213487e-01, -6.926461297460799e-01, -7.092709030981632e-01, -7.092709030981632e-01, -1.573387035116621e-01, -2.283859980866929e-01, -1.534855699648466e-01, -2.331922835865845e+00, -1.642442198872509e-01, -1.642442198872509e-01, -1.023982975504486e-02, -1.169781214807212e-02, -8.768303516628299e-03, -1.125010321327482e-01, -1.065835917371168e-02, -1.065835917371168e-02, -7.371409852377913e-01, -7.259086542656091e-01, -7.298673271811184e-01, -7.329779466970229e-01, -7.314165457929450e-01, -7.314165457929450e-01, -7.147067436985154e-01, -5.744923743781101e-01, -6.114117309994690e-01, -6.490569972172160e-01, -6.296601838389372e-01, -6.296601838389372e-01, -8.027543964350499e-01, -2.696747536345719e-01, -3.091785987807407e-01, -3.975282116048121e-01, -3.476625582751376e-01, -3.476625582751375e-01, -5.165175435997914e-01, -8.191551061357821e-02, -1.051542699543222e-01, -3.811412234311899e-01, -1.327728209339912e-01, -1.327728209339912e-01, -2.575126058005799e-02, -3.207533363030724e-03, -6.090772831351536e-03, -1.264066875638490e-01, -9.060155112956061e-03, -9.060155112956049e-03]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_gga_x_lambda_ch_n_BrOH_1_vsigma():
    # Prepare the input
    inp = test_data["BrOH"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_lambda_ch_n", 1)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [-6.992023514281638e-09, -6.991982529955212e-09, -6.991710796493537e-09, -6.992383699808519e-09, -6.992002643632483e-09, -6.992002643632483e-09, -9.407912689631965e-06, -9.408143432840444e-06, -9.412930194913197e-06, -9.396706175574745e-06, -9.407871061195118e-06, -9.407871061195118e-06, -5.952754595399298e-03, -5.953790701873620e-03, -5.889601513573478e-03, -5.747376316855591e-03, -5.953256662830718e-03, -5.953256662830718e-03, -6.178506799865653e-01, -6.143050455394289e-01, -3.186463951816564e-03, -8.958940837664864e-01, -6.172402495980370e-01, -6.172402495980370e-01, -4.558855916764513e+00, -4.571409930339863e+00, -1.918371005658153e+00, -3.227515197209062e+00, -4.748927059805257e+00, -4.748927059805257e+00, -2.066613882337628e-06, -2.066917685770275e-06, -2.066636185210477e-06, -2.066872567675503e-06, -2.066782501673080e-06, -2.066782501673080e-06, -7.159424842580768e-05, -7.028969523056214e-05, -7.141836031795698e-05, -7.040995328816754e-05, -7.091091703704250e-05, -7.091091703704250e-05, -1.250570441821017e-02, -1.025817595965457e-02, -1.524090765174898e-02, -1.450263601204833e-02, -1.100014102266198e-02, -1.100014102266198e-02, -1.043929830805965e+00, -3.636514344668387e-01, -1.195093276709879e+00, -1.198047134023458e-04, -1.112994446105004e+00, -1.112994446105004e+00, -3.423891068693414e+00, -3.431064454333995e+00, -9.820323780505293e+00, -1.877628500520310e+00, -5.067859899108020e+00, -5.067859899108020e+00, -1.302998938853081e-02, -1.294394811718648e-02, -1.297465618687151e-02, -1.299869464798906e-02, -1.298670910062804e-02, -1.298670910062804e-02, -1.491500481685250e-02, -1.963206461036110e-02, -1.821443684186799e-02, -1.694246203661792e-02, -1.760409926883082e-02, -1.760409926883082e-02, -8.450980282671465e-03, -2.085205275507067e-01, -1.425963195737082e-01, -7.812130930602296e-02, -1.082346613909717e-01, -1.082346613909717e-01, -2.903527276348909e-02, -1.621831767285269e+00, -1.619162091951226e+00, -1.082187721726661e-01, -1.788796897804958e+00, -1.788796897804960e+00, -3.425350450588783e+00, -5.900772838808407e+00, -5.080892663087210e+00, -2.195762636739523e+00, -7.401069089137786e+00, -7.401069089137775e+00]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05