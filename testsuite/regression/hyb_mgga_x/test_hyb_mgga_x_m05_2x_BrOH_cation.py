
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_hyb_mgga_x_m05_2x_BrOH_cation_2_zk():
    # Prepare the input
    inp = test_data["BrOH_cation"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_mgga_x_m05_2x", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-7.828262356721078e+00, -7.828255440007144e+00, -7.828256917851904e+00, -7.828358015981743e+00, -7.828300483528806e+00, -7.828300483528806e+00, -1.592441377017076e+00, -1.592444946253974e+00, -1.592615003159563e+00, -1.593339300523499e+00, -1.592815417675173e+00, -1.592815417675173e+00, -3.532751554815141e-01, -3.541531833941006e-01, -3.760148168238379e-01, -3.690354155764858e-01, -3.701176428982825e-01, -3.701176428982825e-01, -1.257750326443977e-01, -1.247954860823459e-01, -4.788721586398135e-01, -1.340612004636431e-01, -1.343950562786260e-01, -1.343950562786260e-01, -1.668351437834668e-02, -1.756026737996545e-02, -8.871298668145999e-02, -9.644783429383866e-03, -1.344255764123817e-02, -1.344255764123816e-02, -1.798995324273363e+00, -1.796170185495390e+00, -1.798839785736562e+00, -1.796347341345269e+00, -1.797579657296296e+00, -1.797579657296296e+00, -9.564008597374717e-01, -9.628517760785988e-01, -9.561748522773221e-01, -9.611661031100627e-01, -9.604464213394326e-01, -9.604464213394326e-01, -2.460608354152131e-01, -2.242782940432003e-01, -2.473936061765355e-01, -2.264084809880269e-01, -2.399108835312762e-01, -2.399108835312762e-01, -1.488668382394126e-01, -1.475948175095614e-01, -1.462091508734639e-01, -7.258475555910453e-01, -1.285784582449426e-01, -1.285784582449420e-01, -7.449866705015731e-03, -9.429151202034315e-03, -7.216355523318724e-03, -1.225319940736793e-01, -9.062530571933678e-03, -9.062530571933725e-03, -2.601925964365625e-01, -1.995098981876658e-01, -2.160179058855463e-01, -2.357927507367607e-01, -2.253642095900539e-01, -2.253642095900537e-01, -2.438953983805050e-01, -2.299867847525043e-01, -2.068393535353403e-01, -1.941726757705230e-01, -1.977320843599832e-01, -1.977320843599832e-01, -2.393790466275511e-01, -1.634060042547026e-01, -1.658872615652662e-01, -1.653951306176993e-01, -1.569765809878285e-01, -1.569765809878285e-01, -2.163846065595393e-01, -8.549243268251991e-02, -1.094323303614897e-01, -1.567092274429572e-01, -1.274390810078649e-01, -1.274390810078644e-01, -2.347845818483709e-02, -2.522492382937404e-03, -5.302267034787686e-03, -1.243956226441646e-01, -8.324089508695707e-03, -8.324089508695650e-03]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_hyb_mgga_x_m05_2x_BrOH_cation_2_vrho():
    # Prepare the input
    inp = test_data["BrOH_cation"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_mgga_x_m05_2x", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-8.474013798757468e+00, -8.473937464844285e+00, -8.474046013565598e+00, -8.473960582803540e+00, -8.474138449560282e+00, -8.474094127928538e+00, -8.473772733546463e+00, -8.473633492263358e+00, -8.474033291526167e+00, -8.473811606796225e+00, -8.474033291526167e+00, -8.473811606796225e+00, -1.984351441343370e+00, -1.984739301221052e+00, -1.983980955339369e+00, -1.984507975620049e+00, -1.976618615390296e+00, -1.975297415366645e+00, -1.979022478484922e+00, -1.979148616667559e+00, -1.984842558755671e+00, -1.972261802261611e+00, -1.984842558755671e+00, -1.972261802261611e+00, -1.551092766495308e-01, -1.569631325832732e-01, -1.547470665840414e-01, -1.566258860459997e-01, -1.662912490547674e-01, -1.693652285687776e-01, -1.532051466860176e-01, -1.501946167621060e-01, -1.617675609597470e-01, -1.863180985896819e-01, -1.617675609597470e-01, -1.863180985896819e-01, -8.137521697347555e-02, -6.551044667355846e-02, -7.795454883746074e-02, -5.811341098436583e-02, -3.989322639928773e-01, -3.647621560883923e-01, -3.777269787813331e-02, -4.522149566236836e-02, -8.388817126363943e-02, -7.135963073565385e-02, -8.388817126363632e-02, -7.135963073565682e-02, -2.141172141665825e-02, -2.272779143959518e-02, -2.241440309155020e-02, -2.397710324265271e-02, -1.021589821327124e-01, -1.047532386892381e-01, -1.294489614846605e-02, -1.273028105582351e-02, -1.913653429622034e-02, -1.092946915162466e-02, -1.913653429622034e-02, -1.092946915162508e-02, -1.678727815204863e+00, -1.678256816650225e+00, -1.674004428903204e+00, -1.673686317543358e+00, -1.678362159538473e+00, -1.678027166511487e+00, -1.674311818603593e+00, -1.673878000655599e+00, -1.676296203670911e+00, -1.675901398939059e+00, -1.676296203670911e+00, -1.675901398939059e+00, -1.049380539188917e+00, -1.053658871499868e+00, -1.094653350386273e+00, -1.096439559764689e+00, -9.979064312502489e-01, -1.018438249324119e+00, -1.048032494141742e+00, -1.065369278516570e+00, -1.103694392513000e+00, -1.082111202524776e+00, -1.103694392513000e+00, -1.082111202524776e+00, -1.522711061649695e-01, -1.492777633465944e-01, -2.611890048662490e-01, -2.579492175349139e-01, -2.951766864506881e-01, -2.284829202440405e-01, -1.677490321342913e-01, -1.574858308686396e-01, -1.661988945462719e-01, -1.479710978657362e-01, -1.661988945462720e-01, -1.479710978657361e-01, -7.431236233543034e-02, -7.258282396401222e-02, -8.849934350702036e-02, -8.912494519083076e-02, -8.859509839677597e-02, -7.791331935709150e-02, -5.650732625105503e-01, -5.652637629835916e-01, -4.186783846598954e-02, -3.489329442083196e-02, -4.186783846598311e-02, -3.489329442082007e-02, -9.723400038355930e-03, -1.010459103527114e-02, -1.245764281694673e-02, -1.264702355415253e-02, -9.304993684987538e-03, -9.848072726325456e-03, -1.112224932080908e-01, -1.077009981435327e-01, -9.520088858738954e-03, -1.302870151204802e-02, -9.520088858739451e-03, -1.302870151204797e-02, -4.353255433819728e-01, -4.278531340928449e-01, -3.671653149044437e-01, -3.782832740440102e-01, -4.427321145656237e-01, -4.513289039814079e-01, -4.769559313093631e-01, -4.794193458683101e-01, -4.651913445221299e-01, -4.708980321967448e-01, -4.651913445221299e-01, -4.708980321967429e-01, -4.546171850442987e-01, -4.506278585634528e-01, -1.772273271724234e-01, -1.725475572970093e-01, -1.320479084420044e-01, -1.359234370365458e-01, -2.103771599112775e-01, -2.103101238640950e-01, -1.877926805406387e-01, -1.908820333021936e-01, -1.877926805406388e-01, -1.908820333021935e-01, -2.566044462907964e-01, -2.687842523437864e-01, -1.036591421960983e-01, -1.026831171329128e-01, -5.938601936014724e-02, -5.594398050273975e-02, -1.833024540040226e-01, -1.849740699483359e-01, -8.661502034557489e-02, -8.825699071592912e-02, -8.661502034557490e-02, -8.825699071592895e-02, -2.482595260694432e-01, -2.390838866245759e-01, -1.008281898005335e-01, -1.013951779446652e-01, -1.169996485287933e-01, -1.171214920999804e-01, -1.892164194374710e-01, -1.549095404195075e-01, -8.546895996475283e-02, -6.723060029431532e-02, -8.546895996476249e-02, -6.723060029431024e-02, -3.049916080855896e-02, -3.157670697856257e-02, -3.359033296665209e-03, -3.366614790227834e-03, -6.827813954904225e-03, -7.260425312749882e-03, -8.356739725489273e-02, -7.756312483290867e-02, -9.011004929551023e-03, -1.195097733755413e-02, -9.011004929551682e-03, -1.195097733755435e-02]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_hyb_mgga_x_m05_2x_BrOH_cation_2_vsigma():
    # Prepare the input
    inp = test_data["BrOH_cation"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_mgga_x_m05_2x", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [-5.270675038058321e-09, 0.000000000000000e+00, -5.270612947340932e-09, -5.270619090925619e-09, 0.000000000000000e+00, -5.270572307028958e-09, -5.270397508274616e-09, 0.000000000000000e+00, -5.270266736155398e-09, -5.271029537025773e-09, 0.000000000000000e+00, -5.271072211950294e-09, -5.270643475170310e-09, 0.000000000000000e+00, -5.270718957634233e-09, -5.270643475170310e-09, 0.000000000000000e+00, -5.270718957634233e-09, -8.723056217230265e-06, 0.000000000000000e+00, -8.725572311682243e-06, -8.723356037256246e-06, 0.000000000000000e+00, -8.726028455714811e-06, -8.731936244911683e-06, 0.000000000000000e+00, -8.733746339795827e-06, -8.715038016776600e-06, 0.000000000000000e+00, -8.716921423309786e-06, -8.727731655281406e-06, 0.000000000000000e+00, -8.723746483414481e-06, -8.727731655281406e-06, 0.000000000000000e+00, -8.723746483414481e-06, -6.177023398434349e-03, 0.000000000000000e+00, -6.110715990324253e-03, -6.200837335148657e-03, 0.000000000000000e+00, -6.125392670236018e-03, -6.531661143607320e-03, 0.000000000000000e+00, -6.562657897961033e-03, -6.229202436385736e-03, 0.000000000000000e+00, -6.188502375466599e-03, -6.025641569367663e-03, 0.000000000000000e+00, -6.317725598539514e-03, -6.025641569367663e-03, 0.000000000000000e+00, -6.317725598539514e-03, -7.654426389032308e-01, 0.000000000000000e+00, -6.939050042720929e-01, -7.626729654017711e-01, 0.000000000000000e+00, -6.758140346463517e-01, -4.255040900911868e-03, 0.000000000000000e+00, -3.820178096266635e-03, -1.290481499244110e+00, 0.000000000000000e+00, -1.181968261436637e+00, -6.588851157998011e-01, 0.000000000000000e+00, -2.612028481991084e+00, -6.588851157998001e-01, 0.000000000000000e+00, -2.612028481991108e+00, -8.092919228774276e+00, 0.000000000000000e+00, -8.008705331527265e+00, -8.524122103437939e+00, 0.000000000000000e+00, -8.476198945593550e+00, -4.418301500209276e+00, 0.000000000000000e+00, -4.398012386448298e+00, -7.427825065378444e+00, 0.000000000000000e+00, -7.226373119751529e+00, -8.084713905001172e+00, 0.000000000000000e+00, -2.054023629007980e+01, -8.084713905001193e+00, 0.000000000000000e+00, -2.054023629007963e+01, -1.483036247054178e-06, 0.000000000000000e+00, -1.484393106857196e-06, -1.481007232480516e-06, 0.000000000000000e+00, -1.482430929615414e-06, -1.482881857102257e-06, 0.000000000000000e+00, -1.484295548850978e-06, -1.481142442721477e-06, 0.000000000000000e+00, -1.482514551584513e-06, -1.482032318115652e-06, 0.000000000000000e+00, -1.483415910789050e-06, -1.482032318115652e-06, 0.000000000000000e+00, -1.483415910789050e-06, -6.632617688660088e-05, 0.000000000000000e+00, -6.634897771531047e-05, -6.515927552753879e-05, 0.000000000000000e+00, -6.522727897577028e-05, -6.619513200050886e-05, 0.000000000000000e+00, -6.626362376938003e-05, -6.515125400018905e-05, 0.000000000000000e+00, -6.521080992980478e-05, -6.575624974294728e-05, 0.000000000000000e+00, -6.577382538431521e-05, -6.575624974294728e-05, 0.000000000000000e+00, -6.577382538431521e-05, -1.019738477851140e-02, 0.000000000000000e+00, -1.024120893546305e-02, -7.311576642585946e-03, 0.000000000000000e+00, -7.285790370012293e-03, -1.575368904053264e-02, 0.000000000000000e+00, -1.371788379503368e-02, -1.450879260707846e-02, 0.000000000000000e+00, -1.209874961802409e-02, -8.368908483195357e-03, 0.000000000000000e+00, -1.070910684457684e-02, -8.368908483195358e-03, 0.000000000000000e+00, -1.070910684457684e-02, -2.084916702815451e+00, 0.000000000000000e+00, -2.103949036771755e+00, -4.980243190056489e-01, 0.000000000000000e+00, -4.927474148111918e-01, -2.531704565776571e+00, 0.000000000000000e+00, -2.338741375817483e+00, -9.561558429030509e-05, 0.000000000000000e+00, -9.577158876082556e-05, -1.879388184915536e+00, 0.000000000000000e+00, -1.934210658727087e+00, -1.879388184915537e+00, 0.000000000000000e+00, -1.934210658727067e+00, -1.043220355596512e+01, 0.000000000000000e+00, -9.029295614462900e+00, -8.979546580755308e+00, 0.000000000000000e+00, -8.290827316859435e+00, -5.112751540341607e+01, 0.000000000000000e+00, -5.686383035112807e+01, -4.444686871132992e+00, 0.000000000000000e+00, -4.220519453573321e+00, -2.544184097507164e+01, 0.000000000000000e+00, -2.495763319923117e+01, -2.544184097507173e+01, 0.000000000000000e+00, -2.495763319923125e+01, -1.312522185598852e-02, 0.000000000000000e+00, -1.300785862529858e-02, -9.935529203036609e-03, 0.000000000000000e+00, -9.863956861327150e-03, -1.078175049773643e-02, 0.000000000000000e+00, -1.074502563659263e-02, -1.181861877964199e-02, 0.000000000000000e+00, -1.176196361562222e-02, -1.126978189562646e-02, 0.000000000000000e+00, -1.122807980419289e-02, -1.126978189562646e-02, 0.000000000000000e+00, -1.122807980419287e-02, -1.424995778787792e-02, 0.000000000000000e+00, -1.419731485430133e-02, -1.913689893614375e-02, 0.000000000000000e+00, -1.883150133962162e-02, -1.571736653367956e-02, 0.000000000000000e+00, -1.541152705194636e-02, -1.338174206717448e-02, 0.000000000000000e+00, -1.319931212887763e-02, -1.434256330013281e-02, 0.000000000000000e+00, -1.412623477850817e-02, -1.434256330013281e-02, 0.000000000000000e+00, -1.412623477850817e-02, -6.146659767433046e-03, 0.000000000000000e+00, -6.077080529986467e-03, -2.705211265618017e-01, 0.000000000000000e+00, -2.670278375458902e-01, -1.674313705328284e-01, 0.000000000000000e+00, -1.627031861673187e-01, -7.365391457273764e-02, 0.000000000000000e+00, -7.232643091060874e-02, -1.078094668173152e-01, 0.000000000000000e+00, -1.082112955374293e-01, -1.078094668173153e-01, 0.000000000000000e+00, -1.082112955374293e-01, -2.697479329370695e-02, 0.000000000000000e+00, -2.647258020387124e-02, -4.045636757931346e+00, 0.000000000000000e+00, -4.049290000347672e+00, -3.797683406246686e+00, 0.000000000000000e+00, -3.838822822741196e+00, -9.686471959454560e-02, 0.000000000000000e+00, -9.214783886977135e-02, -3.833795930556232e+00, 0.000000000000000e+00, -4.139496943217973e+00, -3.833795930556211e+00, 0.000000000000000e+00, -4.139496943217967e+00, -6.405495346070574e+00, 0.000000000000000e+00, -6.537309551323419e+00, -3.210991489312725e+01, 0.000000000000000e+00, -5.687781473872180e+01, -1.981028123758005e+01, 0.000000000000000e+00, -2.108785021652482e+01, -4.297052595254724e+00, 0.000000000000000e+00, -4.087221713068883e+00, -5.244064061300752e+01, 0.000000000000000e+00, -2.591622789768513e+01, -5.244064061300749e+01, 0.000000000000000e+00, -2.591622789768501e+01]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_hyb_mgga_x_m05_2x_BrOH_cation_2_vlapl():
    # Prepare the input
    inp = test_data["BrOH_cation"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_mgga_x_m05_2x", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vlapl"].flatten()
    ref_tgt = [0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_hyb_mgga_x_m05_2x_BrOH_cation_2_vtau():
    # Prepare the input
    inp = test_data["BrOH_cation"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_mgga_x_m05_2x", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vtau"].flatten()
    ref_tgt = [-5.410494094862651e-04, -5.411261322165026e-04, -5.410578001052989e-04, -5.411323845085156e-04, -5.410815797640496e-04, -5.411684271494002e-04, -5.409885444927672e-04, -5.410464170902232e-04, -5.410548599811347e-04, -5.410918426707475e-04, -5.410548599811347e-04, -5.410918426707475e-04, 1.005240691065945e-03, 1.007468968610114e-03, 1.002397692138283e-03, 1.005617691029338e-03, 9.447205788076683e-04, 9.338992713953371e-04, 9.665262949390683e-04, 9.663458645030443e-04, 1.011706111688163e-03, 9.119929339798311e-04, 1.011706111688163e-03, 9.119929339798311e-04, -2.179135551214336e-02, -2.189472637223428e-02, -2.174379297051588e-02, -2.188889657272712e-02, -1.962887234245649e-02, -1.914643571413628e-02, -2.107373369386365e-02, -2.153086108917675e-02, -2.166337022216612e-02, -1.704829953820750e-02, -2.166337022216612e-02, -1.704829953820750e-02, -3.317717031860751e-02, -5.006472480048228e-02, -3.629852311405285e-02, -5.773214763777085e-02, -4.883917158391418e-03, -7.235125549865089e-03, -5.818160766964216e-02, -5.583848632594820e-02, -3.233124366665734e-02, -4.701506246770377e-02, -3.233124366665659e-02, -4.701506246770690e-02, -6.435726976343409e-05, -7.750137052470712e-05, -7.676164585019732e-05, -9.553765147474460e-05, -5.758106659225434e-03, -7.026211474144878e-03, -1.024921651633877e-05, -9.845398166561701e-06, -4.342450400105896e-05, -1.371106820916583e-05, -4.342450400105896e-05, -1.371106821045822e-05, -7.731040080812923e-03, -7.733642434075818e-03, -7.827602123393868e-03, -7.827077189071878e-03, -7.738082970918818e-03, -7.738040354115783e-03, -7.820891704077623e-03, -7.822869363275456e-03, -7.780408148621508e-03, -7.781308687765541e-03, -7.780408148621508e-03, -7.781308687765541e-03, 1.163271263402635e-03, 1.248784447897970e-03, 1.892543958649131e-03, 1.936786879028091e-03, 3.283978924781473e-04, 6.771926575094078e-04, 1.152446583381988e-03, 1.447757260400244e-03, 2.026016111113531e-03, 1.710141404362108e-03, 2.026016111113531e-03, 1.710141404362108e-03, -5.968419433759915e-02, -6.133396183094680e-02, -1.868881094602232e-02, -2.132722690953323e-02, 3.826477468480508e-03, -2.392138545457700e-02, -5.558439915069551e-02, -6.283483942561324e-02, -5.747459295465247e-02, -6.268428627811015e-02, -5.747459295465247e-02, -6.268428627811017e-02, -5.103937089489294e-02, -5.154439343582522e-02, -2.996851269838135e-02, -2.961144673472279e-02, -4.350728798052379e-02, -4.953324867809142e-02, -1.770888917293087e-02, -1.769987638964576e-02, -6.723788098845515e-02, -7.072782209605905e-02, -6.723788098845733e-02, -7.072782209605438e-02, -3.147691976554573e-06, -3.251544041666676e-06, -8.660289781927873e-06, -7.985679464386630e-06, -2.607216962820051e-05, -3.398912321047534e-05, -2.360220171280281e-02, -2.663273384927542e-02, -5.235408398821996e-06, -4.303673409891832e-05, -5.235408398389783e-06, -4.303673409924773e-05, 3.951322760109431e-01, 3.571379201765241e-01, 2.375482557536426e-01, 2.582580288258014e-01, 4.223243430496165e-01, 4.404788054470121e-01, 5.318972754268827e-01, 5.342392588066517e-01, 4.890891809636214e-01, 5.005542451961958e-01, 4.890891809636214e-01, 5.005542451962015e-01, 5.176618129257181e-01, 4.975198164325573e-01, -3.666268221661807e-02, -3.945063695024970e-02, -7.075817753784248e-02, -6.879664690608321e-02, -2.672621973649469e-02, -2.758570371719796e-02, -3.725834222499917e-02, -3.540184920683903e-02, -3.725834222499917e-02, -3.540184920683912e-02, -2.563345734822153e-02, -1.992543348476926e-02, -2.694440086955969e-02, -2.758690580662591e-02, -5.343791136283816e-02, -5.564367357857544e-02, 4.074756118394649e-03, 4.399314774304957e-03, -4.183308857584914e-02, -4.097281607782774e-02, -4.183308857584899e-02, -4.097281607782780e-02, 6.148553990881317e-03, 1.025043756141166e-03, -5.739675537303549e-03, -5.738510430686019e-03, -1.119456506675593e-02, -1.266729054317326e-02, 1.219621122401193e-02, -2.080888622300249e-02, -4.532440870131321e-02, -5.806534224235497e-02, -4.532440870131225e-02, -5.806534224236446e-02, -9.077062429946922e-05, -9.951031422638745e-05, -1.939780593515213e-07, -1.944505598887037e-07, -4.529272362932245e-06, -5.772705106071942e-06, -4.653489876411533e-02, -5.212454225242892e-02, -1.148154488055332e-05, -3.436175466404435e-05, -1.148154487945986e-05, -3.436175466218435e-05]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05