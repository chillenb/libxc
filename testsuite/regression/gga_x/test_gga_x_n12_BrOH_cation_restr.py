
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_gga_x_n12_BrOH_cation_restr_1_zk():
    # Prepare the input
    inp = test_data["BrOH_cation_restr"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_n12", 1)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-2.076880298904749e+01, -2.076891789587968e+01, -2.076935398138346e+01, -2.076764584197678e+01, -2.076855154644759e+01, -2.076855154644759e+01, -3.469346821599046e+00, -3.469392321917391e+00, -3.470521403304450e+00, -3.469179454232853e+00, -3.469480786949498e+00, -3.469480786949498e+00, -7.005974733511479e-01, -7.003250015684537e-01, -6.951050779050377e-01, -7.003635305182494e-01, -6.986634098100512e-01, -6.986634098100512e-01, -1.916772278134100e-01, -1.930456781161304e-01, -8.169807329118888e-01, -1.375632435852123e-01, -1.648464948870955e-01, -1.648464948870954e-01, -8.303457478609147e-03, -8.644856777190981e-03, -1.415987612478668e-02, -5.223831760872699e-03, -6.372993403291297e-03, -6.372993403291297e-03, -5.243575570164063e+00, -5.245145291876063e+00, -5.243648829782916e+00, -5.245034847925053e+00, -5.244373045371493e+00, -5.244373045371493e+00, -1.943871981877337e+00, -1.957523864398403e+00, -1.932372684570423e+00, -1.943859769157562e+00, -1.957571246427198e+00, -1.957571246427198e+00, -5.838143834356330e-01, -6.069974380236079e-01, -5.404615759576115e-01, -5.386768373241768e-01, -5.906001434615465e-01, -5.906001434615465e-01, -5.091880264114955e-02, -1.948638786043898e-01, -4.150914584297714e-02, -1.898368253898175e+00, -9.521153524590392e-02, -9.521153524590392e-02, -4.136100278138183e-03, -5.119774390963480e-03, -4.017350942795886e-03, -1.745240610248372e-02, -4.757215061405239e-03, -4.757215061405239e-03, -5.508795288246046e-01, -5.556218561768965e-01, -5.540573488941335e-01, -5.526744952088103e-01, -5.533748610701481e-01, -5.533748610701481e-01, -5.332110298322326e-01, -5.011485560313168e-01, -5.131138736172799e-01, -5.231742251780610e-01, -5.181751803760182e-01, -5.181751803760182e-01, -6.392120904488977e-01, -2.500585121987883e-01, -2.864073464671119e-01, -3.443705054927233e-01, -3.108640083161862e-01, -3.108640083161861e-01, -4.583817025419561e-01, -1.424389784155018e-02, -1.203463630274303e-02, -3.226378211931528e-01, -3.712574595533172e-02, -3.712574595533152e-02, -1.073896068578335e-02, -1.481955561052015e-03, -3.014337213901697e-03, -3.088402945983517e-02, -4.435435370904900e-03, -4.435435370904897e-03]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_gga_x_n12_BrOH_cation_restr_1_vrho():
    # Prepare the input
    inp = test_data["BrOH_cation_restr"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_n12", 1)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-3.279273321446245e+01, -3.279264734064958e+01, -3.279258263869201e+01, -3.279385377893672e+01, -3.279314261849414e+01, -3.279314261849414e+01, -5.068887912669551e+00, -5.068729081633274e+00, -5.065190971230103e+00, -5.072405477708912e+00, -5.068833035536291e+00, -5.068833035536291e+00, -8.296022344925620e-01, -8.285657519460023e-01, -7.936759119293332e-01, -8.022225927501070e-01, -8.011787584829624e-01, -8.011787584829624e-01, -1.976691542774719e-01, -1.905637199735880e-01, -9.835869670669047e-01, -2.957542720389184e-01, -2.462568482056201e-01, -2.462568482056208e-01, -1.050051963477878e-02, -1.089939108650622e-02, -2.360961107721306e-02, -6.765080873441904e-03, -8.186007621355844e-03, -8.186007621355851e-03, -7.179799198832997e+00, -7.175378233148339e+00, -7.179627799656934e+00, -7.175723908414023e+00, -7.177535740558135e+00, -7.177535740558135e+00, -2.666785487812790e+00, -2.713337364984627e+00, -2.584148434576099e+00, -2.630165597673555e+00, -2.726834821000462e+00, -2.726834821000462e+00, -7.134164314327252e-01, -7.791174889950593e-01, -6.588937295143771e-01, -6.887078447056849e-01, -7.247691995218124e-01, -7.247691995218124e-01, -2.414892399260451e-01, -3.115385889281006e-01, -2.110397305371869e-01, -2.419957335712832e+00, -2.849942217755053e-01, -2.849942217755053e-01, -5.394268194239933e-03, -6.634872722105942e-03, -5.241736136893078e-03, -8.586250350109145e-02, -6.178891227650475e-03, -6.178891227650475e-03, -7.262726202908545e-01, -7.171936934197857e-01, -7.201205090455155e-01, -7.227488426310594e-01, -7.214089851917823e-01, -7.214089851917823e-01, -7.082090560722241e-01, -5.883374587403757e-01, -6.243784753114613e-01, -6.550266669137967e-01, -6.396294909734257e-01, -6.396294909734257e-01, -8.159602129393501e-01, -2.705324302774938e-01, -2.613961387863261e-01, -3.946728135018144e-01, -3.111032669222048e-01, -3.111032669222039e-01, -5.363193473483687e-01, -2.135154604419405e-02, -3.974458927549204e-02, -3.941867841321041e-01, -1.689228525658072e-01, -1.689228525658083e-01, -1.325380987000617e-02, -1.961789475093411e-03, -3.957406789289891e-03, -1.464311149839891e-01, -5.772684025485378e-03, -5.772684025485386e-03]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_gga_x_n12_BrOH_cation_restr_1_vsigma():
    # Prepare the input
    inp = test_data["BrOH_cation_restr"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_n12", 1)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [1.310711214549590e-08, 1.310670322140315e-08, 1.310527589178518e-08, 1.311135155549410e-08, 1.310811253949681e-08, 1.310811253949681e-08, 8.025387756975178e-06, 8.023106757044530e-06, 7.969653535209896e-06, 8.058551028055803e-06, 8.022183122114586e-06, 8.022183122114586e-06, -4.170145808708754e-03, -4.161709492243911e-03, -4.260509932393980e-03, -4.097469053014000e-03, -4.136674240524589e-03, -4.136674240524589e-03, -4.643615864666996e-01, -5.284558209845855e-01, -2.136762255413330e-03, 1.521483171047659e+00, 2.588404031268871e-01, 2.588404031268915e-01, 5.952589872107372e-01, 8.810347635286271e-01, 9.659888340762963e+00, -1.249440029305804e+00, -7.794361875390616e-01, -7.794361875352765e-01, 8.435527598645936e-07, 8.208761738975636e-07, 8.426158552806586e-07, 8.225966266193006e-07, 8.320101878445379e-07, 8.320101878445379e-07, 1.970032661178604e-05, 2.258275438413952e-05, 1.197390341990708e-05, 1.518438623969867e-05, 2.441955698947356e-05, 2.441955698947356e-05, -1.095054581127497e-02, -1.299309443525642e-02, -1.363011638200705e-02, -1.739170767717579e-02, -1.092445097326764e-02, -1.092445097326764e-02, 6.238295464176169e+00, 2.075985982252449e-01, 7.361342153560955e+00, -1.816060477970030e-04, 4.228019620895182e+00, 4.228019620895182e+00, -2.402638078777200e+00, -1.536585836133465e+00, -1.393276327295955e+01, 1.244377454128832e+01, -5.157956762975367e+00, -5.157956762983183e+00, -1.831354386233538e-02, -1.673510235926508e-02, -1.726774457386280e-02, -1.772736637999398e-02, -1.749553894831693e-02, -1.749553894831693e-02, -2.029019262722570e-02, -1.660723014297540e-02, -1.591390071261045e-02, -1.694647844921915e-02, -1.627450495562087e-02, -1.627450495562087e-02, -1.101154342289716e-02, -1.517653677292392e-01, -1.724810466993794e-01, -6.609492101241356e-02, -1.199475163994435e-01, -1.199475163994438e-01, -2.314159459056483e-02, 8.608332546464876e+00, 1.015326003700156e+01, -6.907178936371354e-02, 1.159092322135044e+01, 1.159092322135051e+01, 1.963670744871265e+00, -1.806960706040130e+01, -6.680986669326394e+00, 1.239999737303860e+01, -7.249322660193904e+00, -7.249322660178235e+00]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05