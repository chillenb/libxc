
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_hyb_mgga_xc_mpwb1k_BrOH_cation_2_zk():
    # Prepare the input
    inp = test_data["BrOH_cation"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_mgga_xc_mpwb1k", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-1.187675497410192e+01, -1.187677015371288e+01, -1.187687575738916e+01, -1.187664909888774e+01, -1.187676244511735e+01, -1.187676244511735e+01, -2.019262401131938e+00, -2.019240769267508e+00, -2.018770196963510e+00, -2.019759615125453e+00, -2.019187609841958e+00, -2.019187609841958e+00, -4.311306223803035e-01, -4.309296230249038e-01, -4.269582522176617e-01, -4.291769465315152e-01, -4.299681560257612e-01, -4.299681560257612e-01, -1.318392982377336e-01, -1.327348898376967e-01, -5.112465427219103e-01, -1.120252985712497e-01, -1.232688394371034e-01, -1.232688394371034e-01, -4.836636508495292e-04, -5.720085897969937e-04, -3.141535376474953e-02, -1.010337708258729e-04, -2.865940858663912e-04, -2.865940858663915e-04, -2.892960121138989e+00, -2.892598655735128e+00, -2.892948405240509e+00, -2.892629259996091e+00, -2.892774196502239e+00, -2.892774196502239e+00, -1.226749899531684e+00, -1.232491542978548e+00, -1.226669321245625e+00, -1.231734408477024e+00, -1.229974906759747e+00, -1.229974906759747e+00, -3.598416303435082e-01, -3.784888879539914e-01, -3.388054827933737e-01, -3.457519733297264e-01, -3.638479810370668e-01, -3.638479810370668e-01, -8.958481179825423e-02, -1.405502941880615e-01, -8.442013042344088e-02, -1.085382325635797e+00, -9.686526272949210e-02, -9.686526272949210e-02, -5.585313040478225e-05, -1.018914497275005e-04, -1.100653425432111e-04, -6.004841405170801e-02, -1.523058175160799e-04, -1.523058175160801e-04, -3.424859162059584e-01, -3.439443409714651e-01, -3.434770406809953e-01, -3.430593716852714e-01, -3.432724557983304e-01, -3.432724557983304e-01, -3.338212409941625e-01, -3.128296386034801e-01, -3.190973633615647e-01, -3.249921119163730e-01, -3.219599610061125e-01, -3.219599610061125e-01, -3.978121905007249e-01, -1.670057652192812e-01, -1.892983327977307e-01, -2.291755743690802e-01, -2.072374612553161e-01, -2.072374612553161e-01, -2.909006404266364e-01, -2.847646567837010e-02, -4.650897414071540e-02, -2.173644698965971e-01, -7.302972642102463e-02, -7.302972642102465e-02, -1.132410461965991e-03, -5.492517103759422e-06, -3.054610069090748e-05, -6.968140733506648e-02, -1.288346487756976e-04, -1.288346487756972e-04]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_hyb_mgga_xc_mpwb1k_BrOH_cation_2_vrho():
    # Prepare the input
    inp = test_data["BrOH_cation"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_mgga_xc_mpwb1k", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-1.423505401540221e+01, -1.423503211070947e+01, -1.423511455428979e+01, -1.423507644685114e+01, -1.423529806004979e+01, -1.423534018071607e+01, -1.423460985508776e+01, -1.423446836755976e+01, -1.423508751995207e+01, -1.423481081592848e+01, -1.423508751995207e+01, -1.423481081592848e+01, -2.377621801249243e+00, -2.377720593038670e+00, -2.377631902439865e+00, -2.377738830524832e+00, -2.377980356672323e+00, -2.378086878529011e+00, -2.377394760334729e+00, -2.377549975080870e+00, -2.377275523790381e+00, -2.377839884982747e+00, -2.377275523790381e+00, -2.377839884982747e+00, -5.081503705779312e-01, -5.100709458364273e-01, -5.070383983271636e-01, -5.094898998126339e-01, -4.887678274178057e-01, -4.849317567644306e-01, -4.882706356693645e-01, -4.889876737055296e-01, -5.068349518255528e-01, -4.727560667622674e-01, -5.068349518255528e-01, -4.727560667622674e-01, -1.353545476332645e-01, -1.343209164416802e-01, -1.376263940804392e-01, -1.365280858307280e-01, -6.214984368746805e-01, -6.307083156419830e-01, -1.058231053110065e-01, -1.054012946191140e-01, -1.159626291896433e-01, -1.229798651498246e-01, -1.159626291896433e-01, -1.229798651498246e-01, -1.625965016340132e-03, -1.910078773546733e-03, -1.892624234758974e-03, -2.276745607634255e-03, -6.971184465633078e-02, -7.241032578444197e-02, -3.824044137954309e-04, -3.608086021778579e-04, -1.180551396382284e-03, -3.806006035537862e-04, -1.180551396382283e-03, -3.806006035537880e-04, -3.558109750160666e+00, -3.557307888651322e+00, -3.559471502508551e+00, -3.558626689804816e+00, -3.558183084044048e+00, -3.557353352182465e+00, -3.559354029258584e+00, -3.558548952203743e+00, -3.558807624396854e+00, -3.557970798546209e+00, -3.558807624396854e+00, -3.557970798546209e+00, -1.312671893700246e+00, -1.312509058687573e+00, -1.322455330796863e+00, -1.322075197675839e+00, -1.302137775089502e+00, -1.304277610857097e+00, -1.310588423705626e+00, -1.312813217655412e+00, -1.325447690355743e+00, -1.319617727050442e+00, -1.325447690355743e+00, -1.319617727050442e+00, -4.389619702764467e-01, -4.380582897551192e-01, -4.784339171898314e-01, -4.783774574097860e-01, -4.095176422047332e-01, -4.145473879637248e-01, -4.280298676164095e-01, -4.345370068275122e-01, -4.510668483249781e-01, -4.400065932475459e-01, -4.510668483249781e-01, -4.400065932475459e-01, -8.740405797867475e-02, -8.669066235289363e-02, -1.342245155613955e-01, -1.342425340587511e-01, -8.507897904554516e-02, -8.375932145966875e-02, -1.371018034629158e+00, -1.370518535378482e+00, -9.289188038788690e-02, -8.747982654414010e-02, -9.289188038788690e-02, -8.747982654414010e-02, -2.007542166945779e-04, -2.093780219235791e-04, -3.739130533256867e-04, -3.762158330834275e-04, -3.612479487030685e-04, -4.429283609426573e-04, -7.736415398807472e-02, -7.802091924049163e-02, -2.830418050590362e-04, -6.686546667301646e-04, -2.830418050590355e-04, -6.686546667301656e-04, -4.431605805105568e-01, -4.441789356843916e-01, -4.397773733721276e-01, -4.407868874773654e-01, -4.410994032914449e-01, -4.421137323667834e-01, -4.420969130922798e-01, -4.431029798628077e-01, -4.416132947739571e-01, -4.426229724482283e-01, -4.416132947739571e-01, -4.426229724482283e-01, -4.318750856497870e-01, -4.327501889873562e-01, -3.622858508856485e-01, -3.630300187676030e-01, -3.825611821412733e-01, -3.834452448815939e-01, -4.032293808697923e-01, -4.040006900593970e-01, -3.928389244986362e-01, -3.935704742407760e-01, -3.928389244986362e-01, -3.935704742407760e-01, -5.009089189228449e-01, -5.009955782435307e-01, -1.661879137476882e-01, -1.662175575447042e-01, -1.990395202435842e-01, -1.991703940837321e-01, -2.677829691214271e-01, -2.684257801061169e-01, -2.301400014872739e-01, -2.300728596776593e-01, -2.301400014872739e-01, -2.300728596776593e-01, -3.386893128380807e-01, -3.393671840995415e-01, -6.851013607982714e-02, -6.892686736539023e-02, -8.182459914118484e-02, -8.163907609798567e-02, -2.623017954552245e-01, -2.612464223326799e-01, -7.763963871553148e-02, -7.235815212955629e-02, -7.763963871553148e-02, -7.235815212955628e-02, -3.916250525951765e-03, -4.346090845562070e-03, -1.753468558566728e-05, -2.263189166361176e-05, -1.002957729886718e-04, -1.220032072455727e-04, -7.440441042589277e-02, -7.425310990331349e-02, -3.343952138326083e-04, -5.345500578740568e-04, -3.343952138326072e-04, -5.345500578740578e-04]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_hyb_mgga_xc_mpwb1k_BrOH_cation_2_vsigma():
    # Prepare the input
    inp = test_data["BrOH_cation"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_mgga_xc_mpwb1k", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [-7.873754880257928e-09, 0.000000000000000e+00, -7.873837271618996e-09, -7.873710476465995e-09, 0.000000000000000e+00, -7.873804591996138e-09, -7.873476781857281e-09, 0.000000000000000e+00, -7.873497496131707e-09, -7.873985964205442e-09, 0.000000000000000e+00, -7.874139930238798e-09, -7.873736762703978e-09, 0.000000000000000e+00, -7.873813830917255e-09, -7.873736762703978e-09, 0.000000000000000e+00, -7.873813830917255e-09, -8.399106199950395e-06, 0.000000000000000e+00, -8.398808167442044e-06, -8.399513480056696e-06, 0.000000000000000e+00, -8.399060889114064e-06, -8.407105757268692e-06, 0.000000000000000e+00, -8.408209279341981e-06, -8.403691286291228e-06, 0.000000000000000e+00, -8.403042474489152e-06, -8.402904940795827e-06, 0.000000000000000e+00, -8.409752385340676e-06, -8.402904940795827e-06, 0.000000000000000e+00, -8.409752385340676e-06, -2.263839793933675e-03, 0.000000000000000e+00, -2.106359012908470e-03, -2.329502587826537e-03, 0.000000000000000e+00, -2.131219739059843e-03, -3.153486845755340e-03, 0.000000000000000e+00, -3.376685677937833e-03, -3.326518774595483e-03, 0.000000000000000e+00, -3.294169357197028e-03, -1.953511516695138e-03, 0.000000000000000e+00, -3.955534252139938e-03, -1.953511516695138e-03, 0.000000000000000e+00, -3.955534252139938e-03, -5.454154405184815e-01, 0.000000000000000e+00, -5.362401180815777e-01, -5.226644976069772e-01, 0.000000000000000e+00, -5.127164784625281e-01, 3.626448938642353e-04, 0.000000000000000e+00, 6.144405559708314e-04, -1.128747339707193e+00, 0.000000000000000e+00, -1.114805539370941e+00, -6.413858116566331e-01, 0.000000000000000e+00, -5.595884366054866e-01, -6.413858116566324e-01, 0.000000000000000e+00, -5.595884366054835e-01, 3.382928325838071e+02, 0.000000000000000e+00, 3.294280259321234e+02, 3.512319814096659e+02, 0.000000000000000e+00, 3.426189268733185e+02, 3.900338686584048e+01, 0.000000000000000e+00, 3.203886599555867e+01, 3.500720906335275e+02, 0.000000000000000e+00, 3.426139511403172e+02, 3.455993124950371e+02, 0.000000000000000e+00, 9.634860828249239e+02, 3.455993124950378e+02, 0.000000000000000e+00, 9.634860828249364e+02, -2.175257530396116e-06, 0.000000000000000e+00, -2.177161021335481e-06, -2.176281014976111e-06, 0.000000000000000e+00, -2.178146426588332e-06, -2.175315670204317e-06, 0.000000000000000e+00, -2.177193161506108e-06, -2.176189615634989e-06, 0.000000000000000e+00, -2.178091758729680e-06, -2.175782775275780e-06, 0.000000000000000e+00, -2.177654965187811e-06, -2.175782775275780e-06, 0.000000000000000e+00, -2.177654965187811e-06, -6.962639364004078e-05, 0.000000000000000e+00, -6.966779721273315e-05, -6.826960780372711e-05, 0.000000000000000e+00, -6.834784637365950e-05, -7.012111798903456e-05, 0.000000000000000e+00, -7.010832218381804e-05, -6.894095741557732e-05, 0.000000000000000e+00, -6.890478503071611e-05, -6.852633072579858e-05, 0.000000000000000e+00, -6.882152855656944e-05, -6.852633072579858e-05, 0.000000000000000e+00, -6.882152855656944e-05, -5.298670836498360e-03, 0.000000000000000e+00, -5.412908482321628e-03, -4.692860322775688e-04, 0.000000000000000e+00, -5.932819262120855e-04, -4.664757721981147e-03, 0.000000000000000e+00, -5.437319455778317e-03, 8.440657255855785e-03, 0.000000000000000e+00, 3.896731838415161e-03, -4.959958721802593e-03, 0.000000000000000e+00, -4.788379212283315e-03, -4.959958721802601e-03, 0.000000000000000e+00, -4.788379212283320e-03, -2.105223651164680e+00, 0.000000000000000e+00, -2.148252358535696e+00, -4.554181763994844e-01, 0.000000000000000e+00, -4.517442140633599e-01, -2.350644112631654e+00, 0.000000000000000e+00, -2.407026277265321e+00, -5.830832841088899e-05, 0.000000000000000e+00, -5.836931773685672e-05, -1.823495354780158e+00, 0.000000000000000e+00, -2.062614624877626e+00, -1.823495354780158e+00, 0.000000000000000e+00, -2.062614624877626e+00, 5.153610048560705e+02, 0.000000000000000e+00, 4.455208667128435e+02, 4.225473543776063e+02, 0.000000000000000e+00, 3.905176493673433e+02, 2.331928907520339e+03, 0.000000000000000e+00, 2.539053475604838e+03, -7.086490360457365e-01, 0.000000000000000e+00, -6.277143909123785e-01, 1.209707726818587e+03, 0.000000000000000e+00, 1.093255357456487e+03, 1.209707726818584e+03, 0.000000000000000e+00, 1.093255357456487e+03, -5.141161985950804e-03, 0.000000000000000e+00, -5.197979488492516e-03, -5.226564190689623e-03, 0.000000000000000e+00, -5.210606422458753e-03, -4.916176455042767e-03, 0.000000000000000e+00, -4.917989176274517e-03, -4.792478514130149e-03, 0.000000000000000e+00, -4.819118557441684e-03, -4.825688623402559e-03, 0.000000000000000e+00, -4.839706848847729e-03, -4.825688623402559e-03, 0.000000000000000e+00, -4.839706848847729e-03, -3.881291137071453e-03, 0.000000000000000e+00, -3.978000481886852e-03, -1.399265457228000e-02, 0.000000000000000e+00, -1.386558737376362e-02, -1.140060321636560e-02, 0.000000000000000e+00, -1.128617428685809e-02, -8.378727298670759e-03, 0.000000000000000e+00, -8.320794211759575e-03, -9.922079678173958e-03, 0.000000000000000e+00, -9.868289858079618e-03, -9.922079678173958e-03, 0.000000000000000e+00, -9.868289858079615e-03, 5.511488674372553e-05, 0.000000000000000e+00, -1.582295845018698e-04, -2.188218409466839e-01, 0.000000000000000e+00, -2.171256690267581e-01, -1.219108713615251e-01, 0.000000000000000e+00, -1.207586641762030e-01, -3.791823168296796e-02, 0.000000000000000e+00, -3.729318647650134e-02, -7.439040328101271e-02, 0.000000000000000e+00, -7.456487544312690e-02, -7.439040328101276e-02, 0.000000000000000e+00, -7.456487544312695e-02, -1.684671419229141e-02, 0.000000000000000e+00, -1.681423056164221e-02, 4.048859219563090e+01, 0.000000000000000e+00, 3.963650036595315e+01, 1.172710382855715e+01, 0.000000000000000e+00, 9.321563218142423e+00, -2.976512916141921e-02, 0.000000000000000e+00, -3.718608815467686e-02, -3.255953705147523e+00, 0.000000000000000e+00, -4.304193097466199e+00, -3.255953705147528e+00, 0.000000000000000e+00, -4.304193097466204e+00, 2.472610553016215e+02, 0.000000000000000e+00, 2.489479083772504e+02, 1.916113000575781e+03, 0.000000000000000e+00, 3.270687997358791e+03, 1.024008731536066e+03, 0.000000000000000e+00, 1.068420559211213e+03, -3.739137879661000e+00, 0.000000000000000e+00, -3.788828480069099e+00, 2.414240872497442e+03, 0.000000000000000e+00, 1.156358313706407e+03, 2.414240872497438e+03, 0.000000000000000e+00, 1.156358313706416e+03]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_hyb_mgga_xc_mpwb1k_BrOH_cation_2_vlapl():
    # Prepare the input
    inp = test_data["BrOH_cation"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_mgga_xc_mpwb1k", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vlapl"].flatten()
    ref_tgt = [0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_hyb_mgga_xc_mpwb1k_BrOH_cation_2_vtau():
    # Prepare the input
    inp = test_data["BrOH_cation"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_mgga_xc_mpwb1k", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vtau"].flatten()
    ref_tgt = [-1.453789028236753e-05, -1.453795539533437e-05, -1.453819599291548e-05, -1.453817725604425e-05, -1.453863452856280e-05, -1.453895670951458e-05, -1.453519611389047e-05, -1.453456982004614e-05, -1.453809985687147e-05, -1.453592645007412e-05, -1.453809985687147e-05, -1.453592645007412e-05, -2.773989440474821e-04, -2.777448485755736e-04, -2.774399024960354e-04, -2.778084148197651e-04, -2.786639438061573e-04, -2.789831013985793e-04, -2.764684431128833e-04, -2.768336125599870e-04, -2.773459159014582e-04, -2.778996350686698e-04, -2.773459159014582e-04, -2.778996350686698e-04, -2.491933376166078e-03, -2.642184406024723e-03, -2.443485038052027e-03, -2.625098956656117e-03, -1.892260784642392e-03, -1.718606872764310e-03, -1.784285750730145e-03, -1.842295697882058e-03, -2.877173658706024e-03, -1.074779912048118e-03, -2.877173658706024e-03, -1.074779912048118e-03, -2.393230896972181e-03, -2.713708004130124e-03, -2.615544232205190e-03, -3.002092192391819e-03, -2.055021085824489e-03, -2.505285064109242e-03, -9.520634906531265e-04, -9.656735293285018e-04, -2.354546618229709e-03, -4.301318033992195e-04, -2.354546618229706e-03, -4.301318033992202e-04, -2.997087148185276e-07, -3.518866688465333e-07, -3.601478287499169e-07, -4.346246878873608e-07, -2.296147427444176e-05, -2.686840224581216e-05, -6.513783696351832e-08, -6.041348501259918e-08, -2.169816319607273e-07, -1.113827375369721e-07, -2.169816319607280e-07, -1.113827375369724e-07, -2.269464053455427e-04, -2.269942041936558e-04, -2.280519143712854e-04, -2.280628082763925e-04, -2.270020733267227e-04, -2.270268297199039e-04, -2.279506377716149e-04, -2.279975920571464e-04, -2.275163759208161e-04, -2.275310559213020e-04, -2.275163759208161e-04, -2.275310559213020e-04, -2.646821730286466e-04, -2.646819876534644e-04, -2.699096506917164e-04, -2.698297537548436e-04, -2.456280887323677e-04, -2.511417099558468e-04, -2.500045673272401e-04, -2.554573505090169e-04, -2.824632236149086e-04, -2.694761276964254e-04, -2.824632236149086e-04, -2.694761276964254e-04, -6.347379240705518e-03, -6.390511160701925e-03, -1.184698105674906e-02, -1.195811997073223e-02, -6.321235538715860e-03, -6.490396976230268e-03, -1.327119950005738e-02, -1.243099175812918e-02, -6.676694104181627e-03, -6.989931597318513e-03, -6.676694104181627e-03, -6.989931597318513e-03, -2.781922350892201e-04, -2.928988132952843e-04, -1.076133359808841e-03, -1.086732839708232e-03, -2.304560833431565e-04, -2.641079765903469e-04, -1.927400997080872e-03, -1.929892281417028e-03, -4.987911163701257e-04, -7.898358953142875e-04, -4.987911163701257e-04, -7.898358953142875e-04, -4.030688190349819e-08, -3.894187119486963e-08, -7.056855839852068e-08, -6.799578383656435e-08, -1.755342139979134e-07, -2.301047862615557e-07, -1.134151689771698e-04, -1.109104482960024e-04, -9.269830704810115e-08, -2.249914142493880e-07, -9.269830704810089e-08, -2.249914142493877e-07, -1.717652751912049e-02, -1.713619248568862e-02, -1.386428745944997e-02, -1.386672723675737e-02, -1.491063312506914e-02, -1.490782769044810e-02, -1.587685030660696e-02, -1.584744786659725e-02, -1.538304991477683e-02, -1.536705133539403e-02, -1.538304991477683e-02, -1.536705133539403e-02, -1.929224223085034e-02, -1.920034085401544e-02, -4.370113379078337e-03, -4.409813643531602e-03, -6.427959367186009e-03, -6.497529364432018e-03, -9.824749734082082e-03, -9.829088996211842e-03, -7.943079478903152e-03, -7.949467297785930e-03, -7.943079478903152e-03, -7.949467297785928e-03, -1.059149578002199e-02, -1.072681605893395e-02, -1.540683307611430e-03, -1.564565636171346e-03, -2.429583102321081e-03, -2.531861492405219e-03, -6.287727115763946e-03, -6.323005669183542e-03, -4.006584648178532e-03, -4.059123384598535e-03, -4.006584648178535e-03, -4.059123384598537e-03, -4.805309699537307e-03, -4.900319655551454e-03, -1.988346384035491e-05, -2.028650905216560e-05, -4.476415823168420e-05, -5.025037174718308e-05, -9.841532611841538e-03, -1.007052461472450e-02, -2.184009360754973e-04, -3.122002109861319e-04, -2.184009360754975e-04, -3.122002109861319e-04, -6.466774595219028e-07, -7.290699924745556e-07, -5.755693045808994e-09, -1.026365461509461e-08, -2.775916785357306e-08, -3.528415037704372e-08, -2.162876966262462e-04, -2.221903081659497e-04, -1.636920504245606e-07, -1.823002569306333e-07, -1.636920504245608e-07, -1.823002569306335e-07]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05