
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_mgga_x_regtm_BrOH_1_zk():
    # Prepare the input
    inp = test_data["BrOH"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_regtm", 1)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-2.226180227951831e+01, -2.226184853329048e+01, -2.226213250216017e+01, -2.226137556105433e+01, -2.226182599738224e+01, -2.226182599738224e+01, -3.403019012574016e+00, -3.403028767074600e+00, -3.403565052227293e+00, -3.405284120709976e+00, -3.403027320589159e+00, -3.403027320589159e+00, -6.689244759444357e-01, -6.687180985775376e-01, -6.642433382779465e-01, -6.679088931669942e-01, -6.688448241528511e-01, -6.688448241528511e-01, -2.054334332420538e-01, -2.067532877684149e-01, -8.320414206711080e-01, -1.581622081590666e-01, -2.057788691622910e-01, -2.057788691622910e-01, -2.594085674599664e-02, -2.666044536513812e-02, -7.005752634422741e-02, -1.689960849831108e-02, -2.641585476642733e-02, -2.641585476642733e-02, -5.410993445525443e+00, -5.411512839811523e+00, -5.411060460344543e+00, -5.411463654255889e+00, -5.411240412868094e+00, -5.411240412868094e+00, -2.104408884510975e+00, -2.120553938754532e+00, -2.104576522636512e+00, -2.117232193207292e+00, -2.115139731152067e+00, -2.115139731152067e+00, -5.825001317337041e-01, -6.131402701674213e-01, -5.419048736471128e-01, -5.465642434749215e-01, -6.099107643786748e-01, -6.099107643786748e-01, -1.274847374871663e-01, -2.178641611740340e-01, -1.250545560925693e-01, -1.807933678757803e+00, -1.411593259437518e-01, -1.411593259437518e-01, -1.612853410825225e-02, -1.755670937202128e-02, -1.332137560660579e-02, -8.774943687912795e-02, -1.623555581075263e-02, -1.623555581075263e-02, -6.163903191652245e-01, -6.138938635401329e-01, -6.147264928406153e-01, -6.154136246846562e-01, -6.150632768169226e-01, -6.150632768169224e-01, -5.940945935373741e-01, -5.330328991846743e-01, -5.491977093641239e-01, -5.648959371570393e-01, -5.565257995869228e-01, -5.565257995869228e-01, -6.315455239586693e-01, -2.659694759736730e-01, -3.031743659528812e-01, -3.604080509699547e-01, -3.318241826473522e-01, -3.318241826473522e-01, -4.715345573717685e-01, -6.806841578320230e-02, -8.258890061022138e-02, -3.331104571968286e-01, -1.061954784444160e-01, -1.061954784444160e-01, -2.900682289721406e-02, -8.060942391409080e-03, -1.143743485969767e-02, -1.010452984236993e-01, -1.423948382015108e-02, -1.423948382015107e-02]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_mgga_x_regtm_BrOH_1_vrho():
    # Prepare the input
    inp = test_data["BrOH"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_regtm", 1)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-2.806586655525448e+01, -2.806597208419448e+01, -2.806649238610027e+01, -2.806476740353955e+01, -2.806592174150530e+01, -2.806592174150530e+01, -4.154034803833365e+00, -4.154083869720995e+00, -4.155632523229563e+00, -4.154211926303973e+00, -4.154075583653249e+00, -4.154075583653249e+00, -7.725180812903207e-01, -7.717283181465586e-01, -7.527914212644559e-01, -7.573333398186026e-01, -7.722307105494618e-01, -7.722307105494618e-01, -2.094394264103286e-01, -2.122046453887915e-01, -9.036414276486454e-01, -1.664214339321941e-01, -2.102389069583341e-01, -2.102389069583341e-01, -1.473932565216075e-02, -1.536645209312931e-02, -5.662851987739166e-02, -1.781631794305302e-02, -1.512946684108729e-02, -1.512946684108727e-02, -6.942959383703865e+00, -6.946004677148671e+00, -6.943285343399346e+00, -6.945652263572165e+00, -6.944501740871903e+00, -6.944501740871903e+00, -2.326334389782553e+00, -2.355759124431018e+00, -2.321881570624517e+00, -2.345212930927399e+00, -2.352021741556004e+00, -2.352021741556004e+00, -7.163860165043362e-01, -8.665080554226000e-01, -6.632533282243795e-01, -7.217923190240312e-01, -7.642720103411433e-01, -7.642720103411433e-01, -1.203875344218164e-01, -2.094020756496689e-01, -1.183782774577839e-01, -2.409429922363810e+00, -1.317255127285105e-01, -1.317255127285105e-01, -1.245855627660627e-02, -1.631791994562002e-02, -8.982815161883703e-03, -9.148098393291629e-02, -1.722501505285493e-02, -1.722501505285495e-02, -8.229386193920609e-01, -8.117326442416709e-01, -8.158562345433993e-01, -8.189712850353490e-01, -8.174226038822900e-01, -8.174226038822897e-01, -7.987290105902006e-01, -6.321897297328890e-01, -6.764839037655879e-01, -7.226570015994400e-01, -6.985221755143514e-01, -6.985221755143516e-01, -8.665799343728134e-01, -2.659677540412444e-01, -3.169107818471302e-01, -4.083790275934674e-01, -3.602544004414077e-01, -3.602544004414079e-01, -5.406004315913927e-01, -4.975385439143190e-02, -7.128973774837785e-02, -3.920928954397250e-01, -9.639775737378962e-02, -9.639775737378978e-02, -3.001773523671150e-02, -9.649402665470264e-03, -7.060667216908657e-03, -9.277672825057777e-02, -1.544912786282711e-02, -1.544912786282710e-02]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_x_regtm_BrOH_1_vsigma():
    # Prepare the input
    inp = test_data["BrOH"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_regtm", 1)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [-7.164271347541437e-09, -7.164185815017674e-09, -7.163392470770559e-09, -7.164772036038790e-09, -7.164230057835730e-09, -7.164230057835730e-09, -1.160950608287706e-05, -1.161267196404793e-05, -1.171134035256267e-05, -1.173572712876898e-05, -1.160973690560945e-05, -1.160973690560945e-05, -2.536082600294088e-03, -2.542424684921020e-03, -2.824533149124184e-03, -3.360798834038014e-03, -2.539097050913239e-03, -2.539097050913239e-03, -8.243385751677413e-01, -8.166017706567884e-01, 5.201811685940781e-05, -6.579583835608994e-01, -8.223866854321319e-01, -8.223866854321319e-01, -1.405309826838997e+03, -1.240076361878662e+03, -2.048947970296483e+01, 7.105596595399562e+02, -1.302599280229934e+03, -1.302599280229935e+03, -1.897832160430004e-06, -1.896807980791485e-06, -1.897573323622721e-06, -1.896786360632476e-06, -1.897516688780308e-06, -1.897516688780308e-06, -9.584281432466577e-05, -9.202973345918549e-05, -9.386903478979829e-05, -9.077948239408433e-05, -9.596822108222532e-05, -9.596822108222532e-05, -2.619735607841879e-02, -2.942747215527905e-02, -2.417346308347803e-02, -1.996481758580572e-02, -2.348761983321779e-02, -2.348761983321779e-02, -2.139395930079988e+00, -5.519173804014824e-01, -2.363769359355340e+00, -1.330278277865471e-04, -2.067645856249308e+00, -2.067645856249308e+00, -3.952903478124020e+03, -9.800340370824289e+02, -1.250385943598496e+04, -2.030290757353814e+00, 9.464054359063164e+02, 9.464054359063323e+02, -1.119166448812501e-02, -1.203582151404346e-02, -1.179357607119366e-02, -1.156954378976841e-02, -1.168699375032693e-02, -1.168699375032695e-02, -1.417560908642304e-02, -2.237529699952195e-02, -2.131574578802799e-02, -1.962716059694379e-02, -2.088750615839155e-02, -2.088750615839156e-02, -1.767392149642447e-02, -2.835180559754704e-01, -1.892104599652581e-01, -1.194342287809538e-01, -1.530801294258687e-01, -1.530801294258687e-01, -4.810850789127014e-02, -2.837964004274317e+01, -9.891874883685061e+00, -1.764707441816596e-01, -4.835080197693803e+00, -4.835080197693787e+00, 2.365159541916911e+01, 6.985315392126922e+04, -2.668388380420094e+04, -5.612645509180263e+00, 2.413220600935909e+03, 2.413220600935894e+03]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_x_regtm_BrOH_1_vlapl():
    # Prepare the input
    inp = test_data["BrOH"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_regtm", 1)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vlapl"].flatten()
    ref_tgt = [0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_x_regtm_BrOH_1_vtau():
    # Prepare the input
    inp = test_data["BrOH"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_regtm", 1)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vtau"].flatten()
    ref_tgt = [7.111568425383537e-04, 7.111557538129777e-04, 7.110963846006288e-04, 7.111113156853280e-04, 7.111567752085057e-04, 7.111567752085057e-04, 2.575135471811805e-03, 2.576638791297632e-03, 2.624814305450933e-03, 2.645269293699016e-03, 2.575289066932521e-03, 2.575289066932521e-03, -4.189438426747716e-03, -4.159248972806118e-03, -2.843409202869448e-03, -1.090411110512285e-03, -4.176704334823344e-03, -4.176704334823344e-03, 4.372050086168559e-02, 4.472527654319145e-02, -9.067239154497071e-03, -4.155769585991707e-04, 4.399500652466264e-02, 4.399500652466264e-02, 1.364697188026976e-02, 1.365903650302240e-02, 1.047358114754943e-02, -4.743900022759388e-03, 1.402726952084838e-02, 1.402726952084838e-02, 2.834938720871118e-03, 2.839924023882009e-03, 2.835160941987071e-03, 2.839054307053682e-03, 2.837879482612121e-03, 2.837879482612121e-03, 6.728273055157052e-03, 6.607831917491385e-03, 6.506901705852090e-03, 6.395392255976156e-03, 6.959956722372903e-03, 6.959956722372903e-03, 5.044276539973018e-02, 8.432100375065714e-02, 2.811971137821804e-02, 2.249491729049634e-02, 5.718935354903043e-02, 5.718935354903043e-02, 1.037310329706881e-02, 3.022771397148763e-02, 1.119044453623146e-02, 4.369190883708284e-03, 2.144058655726498e-02, 2.144058655726498e-02, 9.667817934559095e-04, -2.606875616399238e-03, 4.896050757985421e-03, -7.631505227305614e-03, -5.801653064290838e-03, -5.801653064290858e-03, 3.472345430457466e-02, 3.275319511401910e-02, 3.346255093136610e-02, 3.401947793983287e-02, 3.374251934023438e-02, 3.374251934023438e-02, 4.030052298812561e-02, 2.688924599268539e-02, 3.143909669932774e-02, 3.554246457418256e-02, 3.415764512204279e-02, 3.415764512204257e-02, 4.415576453708353e-02, 3.151732431890895e-02, 3.450429723623737e-02, 4.098029051791725e-02, 4.046593238593794e-02, 4.046593238593793e-02, 4.305420383624747e-02, 1.475667618550288e-02, 8.409976601118797e-03, 4.948507005664721e-02, 1.501477377859015e-02, 1.501477377859010e-02, -7.068266598263509e-03, -4.353854112638299e-03, 4.210494156222590e-03, 1.414130688619624e-02, -6.653434098520347e-03, -6.653434098520305e-03]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05