
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_mgga_x_ktbm_23_BrOH_cation_2_zk():
    # Prepare the input
    inp = test_data["BrOH_cation"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_ktbm_23", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-2.506421409514776e+01, -2.506428127813876e+01, -2.506472868447615e+01, -2.506373284648383e+01, -2.506423343644983e+01, -2.506423343644983e+01, -3.408354566852941e+00, -3.408516919012276e+00, -3.413119990940320e+00, -3.414104361628572e+00, -3.412590946747405e+00, -3.412590946747405e+00, -5.738830752286690e-01, -5.730906768751265e-01, -5.552600286569798e-01, -5.676404084953562e-01, -5.657464345243344e-01, -5.657464345243344e-01, -1.592103184041069e-01, -1.625956720577634e-01, -6.169009279545856e-01, -1.072346852724270e-01, -1.478571438914958e-01, -1.478571438914958e-01, -4.667707165647982e-03, -4.914560101141996e-03, -2.713408650382979e-02, -2.683364534770435e-03, -3.751676913157865e-03, -3.751676913157865e-03, -6.112247040454274e+00, -6.112143671397661e+00, -6.112320165395586e+00, -6.112225785865791e+00, -6.112155905660527e+00, -6.112155905660527e+00, -2.092136255046166e+00, -2.129802899012792e+00, -2.080661533781869e+00, -2.115033420946049e+00, -2.118450132036855e+00, -2.118450132036855e+00, -6.372179157125811e-01, -6.841619968427219e-01, -5.513576027243101e-01, -5.637543455099775e-01, -6.529183190812098e-01, -6.529183190812100e-01, -7.208158575147722e-02, -1.490516409511168e-01, -6.653551233654199e-02, -1.933346291037480e+00, -8.703842525556418e-02, -8.703842525556421e-02, -2.065532352368864e-03, -2.617634154860574e-03, -2.008219843790542e-03, -4.421645211778505e-02, -2.526537606104773e-03, -2.526537606104774e-03, -6.638731523219034e-01, -6.687898059313377e-01, -6.672345261542220e-01, -6.657971180205444e-01, -6.665283077306511e-01, -6.665283077306511e-01, -6.382205364527300e-01, -5.682231976616878e-01, -6.010017349242021e-01, -6.226137447385767e-01, -6.122829101137404e-01, -6.122829101137404e-01, -7.042901242760241e-01, -1.995817596261593e-01, -2.512094387846483e-01, -3.501374557027097e-01, -3.023758758277377e-01, -3.023758758277377e-01, -4.883951325660720e-01, -2.596643897317209e-02, -3.544101453317720e-02, -3.507146586625686e-01, -5.705187838418881e-02, -5.705187838418882e-02, -6.546966300744830e-03, -7.012553190290207e-04, -1.476180681323159e-03, -5.353374876334188e-02, -2.319645456729005e-03, -2.319645456729003e-03]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_mgga_x_ktbm_23_BrOH_cation_2_vrho():
    # Prepare the input
    inp = test_data["BrOH_cation"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_ktbm_23", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-2.948848226345643e+01, -2.948761317455074e+01, -2.948858980567700e+01, -2.948769140266886e+01, -2.948881734713002e+01, -2.948805021459819e+01, -2.948759776664815e+01, -2.948649685002323e+01, -2.948854909524509e+01, -2.948703257444326e+01, -2.948854909524509e+01, -2.948703257444326e+01, -4.983063917432467e+00, -4.982786834950337e+00, -4.983230710823338e+00, -4.982883125897262e+00, -4.986330509365189e+00, -4.986910825498286e+00, -4.986003910071379e+00, -4.985996354078628e+00, -4.981902201633068e+00, -4.988850854400119e+00, -4.981902201633068e+00, -4.988850854400119e+00, -8.339644772917962e-01, -8.409804071947021e-01, -8.317667579301954e-01, -8.397861529761540e-01, -8.007902272390991e-01, -7.954030828431339e-01, -8.172064157560138e-01, -8.221361586066649e-01, -8.472639591704160e-01, -7.834967204143527e-01, -8.472639591704160e-01, -7.834967204143527e-01, -2.124916057123692e-01, -2.300301914602471e-01, -2.162102156795502e-01, -2.363973168201406e-01, -8.270558910834376e-01, -8.773842096382244e-01, -1.421094774967312e-01, -1.469484435250581e-01, -2.215907638351152e-01, -9.932108256110590e-02, -2.215907638351150e-01, -9.932108256110590e-02, -5.876680791310361e-03, -6.242313707520630e-03, -6.161112067569890e-03, -6.596986733340460e-03, -3.460254900124443e-02, -3.654016016001335e-02, -3.553195725077803e-03, -3.490740225924855e-03, -5.249720527400805e-03, -3.015317918432607e-03, -5.249720527400808e-03, -3.015317918432603e-03, -7.223344332551802e+00, -7.221308067052658e+00, -7.227288166215687e+00, -7.225117838435652e+00, -7.223478520642163e+00, -7.221377900235226e+00, -7.226865555559160e+00, -7.224841835059922e+00, -7.225424518789294e+00, -7.223228852352732e+00, -7.225424518789294e+00, -7.223228852352732e+00, -2.819441213355207e+00, -2.820080011535747e+00, -2.843884097581848e+00, -2.843484952680725e+00, -2.802777188977915e+00, -2.808195988927062e+00, -2.824993923648608e+00, -2.830209687476080e+00, -2.845637718139822e+00, -2.834339318235453e+00, -2.845637718139822e+00, -2.834339318235453e+00, -8.487575190634277e-01, -8.454916145228145e-01, -9.468759087035742e-01, -9.469256570477763e-01, -7.665137632269218e-01, -8.019646996974659e-01, -8.232434089284382e-01, -8.534606321282308e-01, -8.778548662293810e-01, -8.408923949100181e-01, -8.778548662293812e-01, -8.408923949100187e-01, -9.517659982492384e-02, -9.612823334214723e-02, -2.029894626173102e-01, -2.040522484372417e-01, -8.554676799535371e-02, -9.052526194386372e-02, -2.877843007554870e+00, -2.876639129737161e+00, -1.112337233175816e-01, -1.206356728301965e-01, -1.112337233175816e-01, -1.206356728301966e-01, -2.698548911878649e-03, -2.801937461460203e-03, -3.437015777752434e-03, -3.492177657695605e-03, -2.557032687049103e-03, -2.708891146942746e-03, -5.823982221501210e-02, -5.856688349430066e-02, -2.654077314709431e-03, -3.567722589232954e-03, -2.654077314709431e-03, -3.567722589232958e-03, -8.712525091598343e-01, -8.749144896131692e-01, -8.462056147489607e-01, -8.499557888880429e-01, -8.545423456982664e-01, -8.583255206994608e-01, -8.618843435180786e-01, -8.655367908968767e-01, -8.581746473101537e-01, -8.618908934088925e-01, -8.581746473101537e-01, -8.618908934088925e-01, -8.566362098858196e-01, -8.593621814775160e-01, -6.942036055864598e-01, -6.965202914432564e-01, -7.149001042983766e-01, -7.176388328637244e-01, -7.537266856084968e-01, -7.565172685461390e-01, -7.317103637981959e-01, -7.340433507933241e-01, -7.317103637981960e-01, -7.340433507933243e-01, -9.952327571862835e-01, -9.963156828962600e-01, -2.762169598358586e-01, -2.783695212962571e-01, -3.503509629668234e-01, -3.559169464425374e-01, -4.927382787930317e-01, -4.953780040239464e-01, -4.240733673392155e-01, -4.242051568330441e-01, -4.240733673392156e-01, -4.242051568330444e-01, -6.504880094008539e-01, -6.546520644345690e-01, -3.384806699579394e-02, -3.409739275119465e-02, -4.585179148501666e-02, -4.741305773084768e-02, -4.778388156140466e-01, -4.866935600411457e-01, -7.300833636221699e-02, -7.761950073800651e-02, -7.300833636221696e-02, -7.761950073800654e-02, -8.506875520906940e-03, -8.821313709108864e-03, -9.384974675320287e-04, -9.442763523293450e-04, -1.865466727450772e-03, -1.984345624007153e-03, -6.991773385580029e-02, -7.122878842797344e-02, -2.507164945032936e-03, -3.270134655396770e-03, -2.507164945032936e-03, -3.270134655396763e-03]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_x_ktbm_23_BrOH_cation_2_vsigma():
    # Prepare the input
    inp = test_data["BrOH_cation"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_ktbm_23", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [-4.772552966463685e-08, 0.000000000000000e+00, -4.772994356010845e-08, -4.772529272555721e-08, 0.000000000000000e+00, -4.772977063026039e-08, -4.772449635006380e-08, 0.000000000000000e+00, -4.772863914267777e-08, -4.772721354105661e-08, 0.000000000000000e+00, -4.773209273954102e-08, -4.772540615946607e-08, 0.000000000000000e+00, -4.773068121842917e-08, -4.772540615946607e-08, 0.000000000000000e+00, -4.773068121842917e-08, -3.419791023407596e-05, 0.000000000000000e+00, -3.418522516896037e-05, -3.420847772817871e-05, 0.000000000000000e+00, -3.419167393957170e-05, -3.440834931090087e-05, 0.000000000000000e+00, -3.444113501071365e-05, -3.434982672909666e-05, 0.000000000000000e+00, -3.434257476375817e-05, -3.419686343887758e-05, 0.000000000000000e+00, -3.452077227183961e-05, -3.419686343887758e-05, 0.000000000000000e+00, -3.452077227183961e-05, -9.150433603191193e-03, 0.000000000000000e+00, -9.397084365401723e-03, -9.097007483034446e-03, 0.000000000000000e+00, -9.361712931994247e-03, -8.244476623935315e-03, 0.000000000000000e+00, -8.223894472710973e-03, -8.869551526064334e-03, 0.000000000000000e+00, -9.093290610855160e-03, -9.317455673477087e-03, 0.000000000000000e+00, -8.206356571526050e-03, -9.317455673477087e-03, 0.000000000000000e+00, -8.206356571526050e-03, -1.131363209342918e+00, 0.000000000000000e+00, -1.322202594999392e+00, -1.152005145109878e+00, 0.000000000000000e+00, -1.361738997003463e+00, -1.586287822021349e-03, 0.000000000000000e+00, -2.042967366119043e-03, -5.720246193520370e-01, 0.000000000000000e+00, -7.444404420898526e-01, -9.590922457864349e-01, 0.000000000000000e+00, -3.778865032761846e-01, -9.590922457864344e-01, 0.000000000000000e+00, -3.778865032761848e-01, -1.234071887001363e+02, 0.000000000000000e+00, -1.117533650689764e+02, -1.120235883827904e+02, 0.000000000000000e+00, -1.008524165036906e+02, -2.308875576780591e+00, 0.000000000000000e+00, -2.299013021165723e+00, -2.045873055826680e+02, 0.000000000000000e+00, -2.270348266870750e+02, -1.405900428968780e+02, 0.000000000000000e+00, -2.688301913489058e+02, -1.405900428968773e+02, 0.000000000000000e+00, -2.688301913489061e+02, -1.445631801867014e-05, 0.000000000000000e+00, -1.447087461123837e-05, -1.445873487585781e-05, 0.000000000000000e+00, -1.447321769297470e-05, -1.445682901918116e-05, 0.000000000000000e+00, -1.447122719685906e-05, -1.445888581220001e-05, 0.000000000000000e+00, -1.447333887505673e-05, -1.445728859872359e-05, 0.000000000000000e+00, -1.447201721345754e-05, -1.445728859872359e-05, 0.000000000000000e+00, -1.447201721345754e-05, -3.093376358011562e-04, 0.000000000000000e+00, -3.112158119451215e-04, -3.160165189404004e-04, 0.000000000000000e+00, -3.176611311241508e-04, -3.057659873773218e-04, 0.000000000000000e+00, -3.086130960520921e-04, -3.121917845317595e-04, 0.000000000000000e+00, -3.150191269900650e-04, -3.156354125298480e-04, 0.000000000000000e+00, -3.151716475241613e-04, -3.156354125298480e-04, 0.000000000000000e+00, -3.151716475241613e-04, -6.303045041135627e-02, 0.000000000000000e+00, -6.455499183072445e-02, -5.561429100421170e-02, 0.000000000000000e+00, -5.614616996831717e-02, -6.475883816394001e-02, 0.000000000000000e+00, -6.688583647638120e-02, -7.046909278192749e-02, 0.000000000000000e+00, -6.787055228705176e-02, -5.946006417227748e-02, 0.000000000000000e+00, -6.702595431260627e-02, -5.946006417227749e-02, 0.000000000000000e+00, -6.702595431260630e-02, -6.528972473960458e-01, 0.000000000000000e+00, -6.351526813417302e-01, -5.224932888682082e-01, 0.000000000000000e+00, -5.157350686292336e-01, -6.642210420766849e-01, 0.000000000000000e+00, -7.081650619954472e-01, -5.569773831814907e-04, 0.000000000000000e+00, -5.581798865578783e-04, -8.326958878356158e-01, 0.000000000000000e+00, -1.154238921978781e+00, -8.326958878356159e-01, 0.000000000000000e+00, -1.154238921978782e+00, 5.002204400851649e+00, 0.000000000000000e+00, -1.835127078063283e+01, -1.271718348114730e+02, 0.000000000000000e+00, -1.014822244108746e+02, -1.019814988557612e+03, 0.000000000000000e+00, -9.308022879696842e+02, -1.025667865324576e+00, 0.000000000000000e+00, -1.317317898691533e+00, 1.777499158530922e+02, 0.000000000000000e+00, -5.553391506941825e+02, 1.777499158530909e+02, 0.000000000000000e+00, -5.553391506941833e+02, -9.283587319217078e-02, 0.000000000000000e+00, -9.148397179233639e-02, -9.349512980608728e-02, 0.000000000000000e+00, -9.215784960680855e-02, -9.333146149133760e-02, 0.000000000000000e+00, -9.198603488081231e-02, -9.314210556427527e-02, 0.000000000000000e+00, -9.179285654332028e-02, -9.324244425493240e-02, 0.000000000000000e+00, -9.189500788787272e-02, -9.324244425493240e-02, 0.000000000000000e+00, -9.189500788787272e-02, -1.018811086587962e-01, 0.000000000000000e+00, -1.006896346346929e-01, -1.224456069108719e-01, 0.000000000000000e+00, -1.214688856468308e-01, -1.241542173411547e-01, 0.000000000000000e+00, -1.228996555109407e-01, -1.193185076941686e-01, 0.000000000000000e+00, -1.178540213909819e-01, -1.224129814430169e-01, 0.000000000000000e+00, -1.211062473564513e-01, -1.224129814430169e-01, 0.000000000000000e+00, -1.211062473564513e-01, -4.301008786627460e-02, 0.000000000000000e+00, -4.373918647199093e-02, -4.319291772365851e-01, 0.000000000000000e+00, -4.306005013269342e-01, -3.742740644658690e-01, 0.000000000000000e+00, -3.793413548459935e-01, -2.895963575241562e-01, 0.000000000000000e+00, -2.837572027589343e-01, -3.639993768677409e-01, 0.000000000000000e+00, -3.678700219086408e-01, -3.639993768677412e-01, 0.000000000000000e+00, -3.678700219086411e-01, -1.338149549102328e-01, 0.000000000000000e+00, -1.360935323945147e-01, -2.965162754408663e+00, 0.000000000000000e+00, -2.819109260765239e+00, -1.380227097232814e+00, 0.000000000000000e+00, -1.375528631819964e+00, -4.195877319330262e-01, 0.000000000000000e+00, -4.754681798756881e-01, -1.165958412711303e+00, 0.000000000000000e+00, -1.366271266575861e+00, -1.165958412711304e+00, 0.000000000000000e+00, -1.366271266575861e+00, -1.629410657091665e+01, 0.000000000000000e+00, -1.358927646079346e+01, 2.667991211908416e+03, 0.000000000000000e+00, 4.211093512153695e+03, -1.575051898495402e+03, 0.000000000000000e+00, -1.422794987640507e+03, -1.402895164386759e+00, 0.000000000000000e+00, -1.658315756589380e+00, 1.077945478464291e+02, 0.000000000000000e+00, -6.711771149753904e+02, 1.077945478464299e+02, 0.000000000000000e+00, -6.711771149753922e+02]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_x_ktbm_23_BrOH_cation_2_vlapl():
    # Prepare the input
    inp = test_data["BrOH_cation"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_ktbm_23", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vlapl"].flatten()
    ref_tgt = [0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_x_ktbm_23_BrOH_cation_2_vtau():
    # Prepare the input
    inp = test_data["BrOH_cation"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_ktbm_23", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vtau"].flatten()
    ref_tgt = [3.221579612284294e-03, 3.221729256542241e-03, 3.221548499500509e-03, 3.221706639761358e-03, 3.221495625870126e-03, 3.221617738226658e-03, 3.221847228406197e-03, 3.222066044885849e-03, 3.221559264091985e-03, 3.221921089354879e-03, 3.221559264091985e-03, 3.221921089354879e-03, 1.133279144756810e-02, 1.132214923775191e-02, 1.133654299833669e-02, 1.132384885447849e-02, 1.140071445343273e-02, 1.140990389252627e-02, 1.141192168497187e-02, 1.140413070464657e-02, 1.132705132822343e-02, 1.146233222936476e-02, 1.132705132822343e-02, 1.146233222936476e-02, 1.902066887735050e-02, 1.959888462431160e-02, 1.889717591949653e-02, 1.949308509916586e-02, 1.676922304954104e-02, 1.693700102987049e-02, 1.942644232643542e-02, 2.003936620161673e-02, 1.926415894128653e-02, 1.871730281379448e-02, 1.926415894128653e-02, 1.871730281379448e-02, 7.129148830883583e-02, 9.852956786932190e-02, 7.432836923418695e-02, 1.059491017326914e-01, 2.522394449932304e-03, 3.990477712242211e-03, 1.297188791244247e-02, 1.963039124993275e-02, 6.777379972075454e-02, 2.760144649848945e-03, 6.777379972075462e-02, 2.760144649848882e-03, 1.939316262328664e-04, 2.123339396712497e-04, 2.011001707466276e-04, 2.247524614953033e-04, 7.389076618295980e-04, 8.904047750229206e-04, 6.222662175875747e-05, 6.689595408162631e-05, 1.528515693731811e-04, 4.420263762767631e-05, 1.528515693731830e-04, 4.420263762767455e-05, 1.239289593441012e-02, 1.239708567758770e-02, 1.236666147982323e-02, 1.237172089628533e-02, 1.239163246685685e-02, 1.239635327257913e-02, 1.236911514087598e-02, 1.237330470324175e-02, 1.237931808459521e-02, 1.238432723286234e-02, 1.237931808459521e-02, 1.238432723286234e-02, 2.628870572893650e-02, 2.647938788600296e-02, 2.742051343738460e-02, 2.757186821644359e-02, 2.612402118092713e-02, 2.635150014461744e-02, 2.717937143840904e-02, 2.741488744629205e-02, 2.703380042359331e-02, 2.708635545120318e-02, 2.703380042359331e-02, 2.708635545120318e-02, 9.970537251111884e-02, 1.011343492097916e-01, 8.294845820420728e-02, 8.312634321906830e-02, 7.865917521591423e-02, 9.007846013630016e-02, 7.588977754465152e-02, 8.084791827357721e-02, 1.004831948545159e-01, 9.947544588173354e-02, 1.004831948545159e-01, 9.947544588173356e-02, 4.991844308081109e-03, 4.952792365597414e-03, 3.457546211838471e-02, 3.453462177916941e-02, 3.582393639397620e-03, 4.643668480931974e-03, 2.373004622476574e-02, 2.374450847711997e-02, 1.033905152394797e-02, 1.801648665359484e-02, 1.033905152394797e-02, 1.801648665359500e-02, -4.320456649767749e-07, 1.989710885936546e-06, 3.055463903957402e-05, 2.493018735020669e-05, 1.151787764822954e-04, 1.241195682390422e-04, 1.648114923533584e-03, 2.281625015288127e-03, -1.308220574786826e-05, 1.916715339650348e-04, -1.308220574786713e-05, 1.916715339650338e-04, 8.395369102743487e-02, 8.345062751804905e-02, 9.338376040904330e-02, 9.274398043189490e-02, 9.011531250679544e-02, 8.950009535850109e-02, 8.734596461340596e-02, 8.681242731461379e-02, 8.873375707899980e-02, 8.816017172050372e-02, 8.873375707899980e-02, 8.816017172050372e-02, 8.449451059033827e-02, 8.412987861792143e-02, 1.445921548219512e-01, 1.444643200133920e-01, 1.395457727341892e-01, 1.388315690308364e-01, 1.209619787491679e-01, 1.203463005961611e-01, 1.312556885329537e-01, 1.307848186890437e-01, 1.312556885329538e-01, 1.307848186890437e-01, 7.665187296583717e-02, 7.736340420664103e-02, 6.027930847855451e-02, 6.108583994757873e-02, 8.535504030197152e-02, 8.895700448382844e-02, 1.189480226080271e-01, 1.179052757365031e-01, 1.168255369435561e-01, 1.177140993375877e-01, 1.168255369435562e-01, 1.177140993375878e-01, 1.218973095535029e-01, 1.259081124817138e-01, 9.437361786875727e-04, 9.085169609425516e-04, 1.067187396768550e-03, 1.191323648425034e-03, 1.381429546294817e-01, 1.653527472152716e-01, 4.032675082705117e-03, 5.759618484644805e-03, 4.032675082705111e-03, 5.759618484644817e-03, 5.924222794368506e-05, 5.419067891383219e-05, -7.054251437057474e-06, -8.405350926253749e-06, 7.444148949908217e-05, 8.072195899935519e-05, 4.324903898063469e-03, 5.680747613714748e-03, -7.466712848512607e-06, 1.781563580057907e-04, -7.466712848512762e-06, 1.781563580057840e-04]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05