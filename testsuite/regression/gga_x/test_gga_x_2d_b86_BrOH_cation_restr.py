
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_gga_x_2d_b86_BrOH_cation_restr_1_zk():
    # Prepare the input
    inp = test_data["BrOH_cation_restr"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_2d_b86", 1)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-1.481541253940627e+02, -1.481547376102614e+02, -1.481576615557061e+02, -1.481485488418732e+02, -1.481532949502280e+02, -1.481532949502280e+02, -9.847147716464606e+00, -9.847152832711643e+00, -9.847630184128343e+00, -9.849581045873389e+00, -9.847506789205370e+00, -9.847506789205370e+00, -9.183912915683562e-01, -9.181479121709816e-01, -9.173804723989885e-01, -9.260264511403278e-01, -9.227963052386798e-01, -9.227963052386798e-01, -2.914312907905783e-01, -2.843943718876258e-01, -1.128266869211167e+00, -3.628446948453083e-01, -3.229933734845547e-01, -3.229933734845547e-01, -1.240495986584289e-02, -1.339456686159708e-02, -1.660320610901110e-01, -5.445820245812760e-03, -7.670393054640668e-03, -7.670393054640668e-03, -1.788572035697724e+01, -1.788996160589931e+01, -1.788593725120115e+01, -1.788968115077013e+01, -1.788785834954537e+01, -1.788785834954537e+01, -4.340132252116200e+00, -4.380248272623032e+00, -4.319713177543600e+00, -4.354884540376522e+00, -4.372366472215460e+00, -4.372366472215460e+00, -7.191634901216859e-01, -7.754183791813486e-01, -6.495712541171443e-01, -6.519093251150390e-01, -7.309965334933475e-01, -7.309965334933475e-01, -4.122047936894075e-01, -3.882271526346229e-01, -3.983148221575916e-01, -4.006491489365672e+00, -3.737111204944243e-01, -3.737111204944243e-01, -3.690918979456406e-03, -5.263772247745275e-03, -3.519545683637492e-03, -2.942896824060384e-01, -4.653657253596923e-03, -4.653657253596924e-03, -6.825770586835970e-01, -6.842895466919379e-01, -6.836762710946739e-01, -6.831700285456985e-01, -6.834209728490059e-01, -6.834209728490059e-01, -6.529259602079098e-01, -5.967317345617780e-01, -6.076684598713876e-01, -6.223242746710373e-01, -6.140621240651312e-01, -6.140621240651312e-01, -8.330073313485615e-01, -3.807307258865987e-01, -3.685530719509982e-01, -3.843909464241541e-01, -3.649013815485118e-01, -3.649013815485117e-01, -5.351389613417660e-01, -1.566079736419365e-01, -2.355313495725307e-01, -3.449132113161500e-01, -3.425936496735328e-01, -3.425936496735327e-01, -2.078892470805268e-02, -7.275016433288576e-04, -2.212385922522465e-03, -3.318068368283356e-01, -4.142196942546623e-03, -4.142196942546617e-03]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_gga_x_2d_b86_BrOH_cation_restr_1_vrho():
    # Prepare the input
    inp = test_data["BrOH_cation_restr"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_2d_b86", 1)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-2.215328492392457e+02, -2.215337873128749e+02, -2.215382274851106e+02, -2.215242652636372e+02, -2.215315428071206e+02, -2.215315428071206e+02, -1.438157848527901e+01, -1.438166506596452e+01, -1.438419917427401e+01, -1.438392599536444e+01, -1.438220699487645e+01, -1.438220699487645e+01, -1.093847224652971e+00, -1.090179572498497e+00, -9.954738311780729e-01, -1.011932213370050e+00, -1.010429671052061e+00, -1.010429671052061e+00, 5.713528355101832e-02, 4.596396929780243e-02, -1.418462835768626e+00, 1.162828720202618e-01, 1.096984675149211e-01, 1.096984675149209e-01, -1.859645413591582e-02, -2.007763161839075e-02, -2.272849813965901e-01, -8.167845345781220e-03, -1.150280107773862e-02, -1.150280107773862e-02, -2.657792064046797e+01, -2.658626540271144e+01, -2.657832774549525e+01, -2.658569428847222e+01, -2.658213838473283e+01, -2.658213838473283e+01, -5.799719331538938e+00, -5.873575300453167e+00, -5.732999904967439e+00, -5.798339242364800e+00, -5.872661601176156e+00, -5.872661601176156e+00, -9.421192618423840e-01, -1.124475314857040e+00, -8.246167840637274e-01, -9.280528817064748e-01, -9.733082495975462e-01, -9.733082495975462e-01, -1.039259327646641e-01, 1.274215215981439e-01, -1.517540566523317e-01, -5.946397712250550e+00, 5.004384238617422e-02, 5.004384238617422e-02, -5.536062244204985e-03, -7.894791336471535e-03, -5.278668872497374e-03, -2.847237221378235e-01, -6.979461408871701e-03, -6.979461408871703e-03, -1.011177671093927e+00, -9.911561498342949e-01, -9.982081347899766e-01, -1.004025835313263e+00, -1.001116631745945e+00, -1.001116631745945e+00, -9.734695054597923e-01, -6.602654537114744e-01, -7.537575691204895e-01, -8.438931462077147e-01, -7.986476910836846e-01, -7.986476910836846e-01, -1.206059008952242e+00, 4.964999383930885e-02, -7.869329562058727e-02, -3.483206966735677e-01, -2.095761047418291e-01, -2.095761047418291e-01, -5.716937430704037e-01, -2.176558054535438e-01, -2.900776271475550e-01, -3.565756867864419e-01, -1.794859895340825e-01, -1.794859895340819e-01, -3.113690226520778e-02, -1.091247396345254e-03, -3.318479697954488e-03, -2.135119957962428e-01, -6.212483495246882e-03, -6.212483495246872e-03]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_gga_x_2d_b86_BrOH_cation_restr_1_vsigma():
    # Prepare the input
    inp = test_data["BrOH_cation_restr"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_2d_b86", 1)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [-1.571629469441060e-09, -1.571609755394230e-09, -1.571516014350568e-09, -1.571809456364785e-09, -1.571656557818400e-09, -1.571656557818400e-09, -5.535897964459754e-06, -5.535846024396193e-06, -5.534035649139880e-06, -5.532455922124453e-06, -5.535235434567529e-06, -5.535235434567529e-06, -8.995866726862053e-03, -9.038941667961488e-03, -1.014753058613043e-02, -9.781138758003211e-03, -9.859764249618912e-03, -9.859764249618912e-03, -3.258219097870522e+00, -3.095645183854536e+00, -4.521013760198952e-03, -7.965808451211559e+00, -5.735847261130493e+00, -5.735847261130492e+00, -1.442943964029570e+00, -1.646551011675021e+00, -1.159956952298343e+01, -5.747352473636238e-01, -1.029072520918197e+00, -1.029072520918807e+00, -9.012871128962817e-07, -9.005470466063326e-07, -9.012502016802723e-07, -9.005968969323201e-07, -9.009134039394942e-07, -9.009134039394942e-07, -7.317530363868390e-05, -7.084848372976840e-05, -7.491367172138650e-05, -7.279338046532932e-05, -7.107235304689066e-05, -7.107235304689066e-05, -1.652324305799462e-02, -1.145217276004486e-02, -2.338803740437822e-02, -1.978182527373631e-02, -1.539092887116653e-02, -1.539092887116653e-02, -1.540281190559080e+01, -2.863997399564200e+00, -1.727396833444489e+01, -8.032340562466631e-05, -1.408693602108442e+01, -1.408693602108442e+01, -5.131296556031826e-01, -6.538262093540422e-01, -2.746603030561370e+00, -2.215190171594762e+01, -1.689970300731011e+00, -1.689970300732785e+00, -1.628716987099211e-02, -1.669173891078642e-02, -1.654803427947855e-02, -1.643079808199779e-02, -1.648936091491445e-02, -1.648936091491445e-02, -1.843821266774635e-02, -3.599409304849208e-02, -2.945612874038665e-02, -2.432828094929955e-02, -2.680203637989808e-02, -2.680203637989808e-02, -9.258343534988217e-03, -1.163521762958639e+00, -5.553053686622959e-01, -1.703464549161098e-01, -3.111365682892588e-01, -3.111365682892589e-01, -5.210806825102380e-02, -9.974351774554046e+00, -1.478024056757634e+01, -2.024825991454329e-01, -2.598531932130956e+01, -2.598531932130955e+01, -1.951965393915104e+00, -4.431538457812381e-01, -6.524078581803880e-01, -2.625802772789784e+01, -1.899907930733913e+00, -1.899907930736730e+00]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05