
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_mgga_c_m11_BrOH_cation_restr_1_zk():
    # Prepare the input
    inp = test_data["BrOH_cation_restr"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_m11", 1)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-3.473275582387698e-01, -3.473454435106487e-01, -3.474202293780576e-01, -3.471555056604940e-01, -3.472948827161908e-01, -3.472948827161908e-01, -6.410855910766504e-02, -6.412466650843032e-02, -6.452919872885067e-02, -6.428917006334924e-02, -6.434758143856185e-02, -6.434758143856185e-02, 6.140090879622776e-03, 6.587385876075312e-03, 1.531041296103351e-02, 1.680253344039745e-02, 1.612840073786558e-02, 1.612840073786558e-02, 1.361757168393167e-02, 1.477064646255182e-02, -3.413413021261417e-02, -1.685568215414985e-02, -3.668768869906086e-03, -3.668768869906089e-03, -3.198751662556857e-02, -3.336547265800057e-02, -1.039458634624411e-01, -2.003637184373397e-02, -2.440342345458939e-02, -2.440342345458939e-02, -2.770029143871844e-01, -2.758617127449262e-01, -2.769525538574709e-01, -2.759454231104104e-01, -2.764355068628291e-01, -2.764355068628291e-01, -2.688318619985403e-02, -3.289909258472586e-02, -2.120134702858408e-02, -2.714238701391883e-02, -3.278665673031635e-02, -3.278665673031635e-02, -4.595059626791572e-02, -5.810731198987647e-02, -4.461061820938590e-02, -5.412873797479462e-02, -4.951044339500129e-02, -4.951044339500129e-02, -6.698023232872367e-02, -3.758421921645695e-04, -7.540115873918507e-02, -7.992595434691425e-02, -2.714061247483237e-02, -2.714061247483214e-02, -1.594620445808262e-02, -1.964573223943340e-02, -1.546713790593422e-02, -1.009707202338711e-01, -1.825565328592321e-02, -1.825565328592322e-02, -2.520485914383104e-01, -8.470472201928865e-02, -1.242664381730138e-01, -1.754334971211986e-01, -1.478243353169111e-01, -1.478243353169111e-01, -2.302126797568116e-01, -3.727714554523685e-02, -4.835034088602741e-02, -7.996046618043728e-02, -6.620969390312559e-02, -6.620969390312557e-02, -6.034829512800315e-02, 1.521274035324838e-02, 2.289491774951360e-02, -2.181107646619919e-02, 4.743183854321716e-03, 4.743183854321644e-03, -3.685157952163886e-02, -1.023509501795326e-01, -1.104062247700471e-01, -3.552866667717472e-02, -7.178602729700415e-02, -7.178602729700326e-02, -4.250370282322631e-02, -5.933008376155013e-03, -1.173203941876276e-02, -7.513247312318116e-02, -1.704173994747578e-02, -1.704173994747574e-02]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_mgga_c_m11_BrOH_cation_restr_1_vrho():
    # Prepare the input
    inp = test_data["BrOH_cation_restr"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_m11", 1)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-1.255373942423535e+00, -1.255336340527837e+00, -1.255199792042303e+00, -1.255756257602621e+00, -1.255460712769865e+00, -1.255460712769865e+00, -1.878664760091890e-01, -1.876674874070251e-01, -1.824274620257559e-01, -1.844574508084438e-01, -1.842723583128880e-01, -1.842723583128880e-01, -8.968535195079139e-02, -8.725600291869905e-02, -2.689820481791417e-02, -5.379213194725228e-02, -4.743482676923061e-02, -4.743482676923061e-02, 3.916398586054631e-02, 3.418843982437875e-02, 1.336737395906340e-02, 3.362138544983816e-02, 2.031906727166615e-02, 2.031906727165897e-02, -4.058235728521058e-02, -4.223540236831946e-02, -9.474809694063605e-02, -2.582598920105093e-02, -3.127206247558431e-02, -3.127206247558430e-02, 1.836685417733867e-02, 3.629527346816606e-02, 1.939433558891117e-02, 3.521204216026474e-02, 2.737009014125247e-02, 2.737009014125247e-02, -3.059574699874854e-01, -2.826027684845019e-01, -3.216868890087765e-01, -3.044367528699121e-01, -2.836942807421046e-01, -2.836942807421046e-01, -9.805507506786620e-02, -3.918505017613722e-03, -8.032849201858953e-02, -7.268794802942792e-02, -1.286213877120117e-01, -1.286213877120117e-01, 5.296128667572473e-02, 3.058083292958120e-02, 4.075860413791781e-02, -1.445598313791366e-01, 5.406329533615523e-02, 5.406329533617354e-02, -2.065429277211079e-02, -2.533786104588769e-02, -1.998369881342641e-02, -3.221821032860162e-02, -2.354067408883884e-02, -2.354067408884000e-02, -6.280603559134414e-01, -3.609069351356298e-01, -5.471858176673216e-01, -6.533642935367739e-01, -6.096666522572435e-01, -6.096666522572435e-01, -6.199255984098142e-01, -4.784218624642040e-02, -1.681655007301492e-01, -7.694557822237080e-02, -1.929056452761335e-01, -1.929056452761335e-01, -5.663409347305020e-02, 5.336903674598482e-02, -1.506433563211413e-02, -1.547315890307183e-01, -1.482374120663418e-01, -1.482374120663414e-01, -9.085740899756498e-02, -9.595427228417305e-02, -7.774938386686880e-02, -7.873415487941013e-02, 3.378431976068714e-02, 3.378431976069820e-02, -5.337274559905870e-02, -7.777875694416190e-03, -1.525469585621408e-02, 2.618270497174721e-02, -2.199573997326503e-02, -2.199573997326227e-02]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_c_m11_BrOH_cation_restr_1_vsigma():
    # Prepare the input
    inp = test_data["BrOH_cation_restr"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_m11", 1)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [-4.364964763340687e-10, -4.365468231741984e-10, -4.367292863495439e-10, -4.359840182102207e-10, -4.363803154718863e-10, -4.363803154718863e-10, 5.446284483375027e-07, 5.447232018103782e-07, 5.469957094841358e-07, 5.421567650752584e-07, 5.444707830699531e-07, 5.444707830699531e-07, 3.655718137011174e-03, 3.651623083338264e-03, 3.328285171333461e-03, 3.099081922762354e-03, 3.194941536118461e-03, 3.194941536118461e-03, 4.099599994961897e-01, 4.245538640521783e-01, 2.653966326646321e-03, 5.622823310319737e-01, 5.134884537717286e-01, 5.134884537717265e-01, 4.816248376848987e-02, 5.673319953770948e-02, 2.801170175220444e-01, 1.703486800487700e-02, 3.559324652545824e-02, 3.559324653741357e-02, -1.082704908077042e-06, -1.088415996875900e-06, -1.082923403404789e-06, -1.087966426901279e-06, -1.085695219794764e-06, -1.085695219794764e-06, 4.217376061202119e-06, 3.813550538118201e-06, 4.356888238432652e-06, 3.955864888281587e-06, 3.932562335246430e-06, 3.932562335246430e-06, 4.005855669222561e-03, -2.004872037294576e-02, 4.274322373332863e-03, 7.539968426379169e-03, 2.879998703018416e-03, 2.879998703018416e-03, 5.190643087298628e-01, 1.959652194537113e-01, 5.544349784354260e-01, 1.715098573829046e-05, 7.375271810437652e-01, 7.375271810437649e-01, 1.664200230172512e-02, 2.094339441829872e-02, 2.095037686334262e-01, 6.244337225594669e-01, 9.118211920960227e-02, 9.118211925249681e-02, 3.505965204888400e-02, 1.606955494591408e-02, 3.950983217285059e-02, 5.244397407388519e-02, 4.749731094136064e-02, 4.749731094136064e-02, 5.916162339143582e-02, 5.349535737719339e-03, 2.546247883398475e-03, -3.143836495939265e-02, -9.351172411209235e-03, -9.351172411209223e-03, -7.596336499371107e-03, 1.145921358338836e-01, 8.944646819630596e-02, 2.531898210586091e-02, 5.646276184339596e-02, 5.646276184339594e-02, 5.820460210742396e-03, 2.297242843063506e-01, 3.420525257995424e-01, 2.852510865397870e-02, 9.577971336892410e-01, 9.577971336892486e-01, 5.979593251716826e-02, 2.271323449544078e-02, 2.857411491330068e-02, 8.964220289734360e-01, 1.135414956055410e-01, 1.135414954900306e-01]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_c_m11_BrOH_cation_restr_1_vlapl():
    # Prepare the input
    inp = test_data["BrOH_cation_restr"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_m11", 1)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vlapl"].flatten()
    ref_tgt = [0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_c_m11_BrOH_cation_restr_1_vtau():
    # Prepare the input
    inp = test_data["BrOH_cation_restr"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_m11", 1)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vtau"].flatten()
    ref_tgt = [6.168605970678904e-04, 6.168527150685872e-04, 6.168257359806705e-04, 6.169425512739259e-04, 6.168804073444234e-04, 6.168804073444234e-04, 6.455716503396878e-04, 6.441156527283330e-04, 6.055551855860457e-04, 6.211401694330409e-04, 6.195560521686444e-04, 6.195560521686444e-04, -6.837196000417336e-05, -3.055022645534111e-04, -5.329345328820877e-03, -2.601337124246913e-03, -3.332474064711661e-03, -3.332474064711661e-03, -6.985567832390281e-02, -6.769817859782161e-02, -7.588002668407966e-03, -5.810582357717579e-02, -5.210610959575766e-02, -5.210610959575523e-02, -3.428596927211816e-04, -4.129630329942137e-04, -1.955126768455585e-02, -5.247954244366029e-05, -1.405679619525514e-04, -1.405679619525514e-04, -9.336943133826448e-04, -1.190643630742347e-03, -9.484601680690781e-04, -1.175146684181144e-03, -1.062077250631744e-03, -1.062077250631744e-03, 4.653981274275815e-03, 4.246803631671292e-03, 4.926260169923539e-03, 4.648157873074239e-03, 4.259192494698067e-03, 4.259192494698067e-03, 8.109340869746573e-03, -3.813775177215510e-03, 2.546980023293991e-03, -1.854352512801343e-03, 2.505137263559828e-02, 2.505137263559828e-02, -8.055303259694557e-02, -4.046814074198361e-02, -7.934497354432375e-02, 2.196282537793419e-03, -8.124235753941839e-02, -8.124235753942816e-02, -1.720442825942150e-05, -4.346814861213827e-05, -1.618504488823943e-04, -5.833495418844377e-02, -1.063840905201588e-04, -1.063840905176444e-04, 1.360331936245181e+00, 5.077721244506166e-01, 9.144018547379086e-01, 1.256825420020005e+00, 1.094010434591840e+00, 1.094010434591840e+00, 1.354899967274209e+00, -1.172317771760176e-02, 6.810201981503884e-02, 8.434096542302588e-02, 1.232218834935550e-01, 1.232218834935550e-01, 9.644457035887528e-03, -4.706892207033053e-02, -1.279172801649976e-02, 6.058243585058574e-02, 6.035997684454385e-02, 6.035997684454397e-02, 1.271111465370897e-02, -1.786050168022772e-02, -3.220617064412397e-02, 1.672393422185908e-02, -9.485064020569996e-02, -9.485064020569997e-02, -4.339268178771963e-04, -1.147949764245056e-06, -2.868349324033059e-05, -9.443339393975445e-02, -1.211876204790995e-04, -1.211876204767355e-04]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05