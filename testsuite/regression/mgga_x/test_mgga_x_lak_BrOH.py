
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_mgga_x_lak_BrOH_1_zk():
    # Prepare the input
    inp = test_data["BrOH"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_lak", 1)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-2.320367804923348e+01, -2.320373743784226e+01, -2.320406447510647e+01, -2.320309263010285e+01, -2.320370882526831e+01, -2.320370882526831e+01, -3.321658813348134e+00, -3.321834666893668e+00, -3.327430373524638e+00, -3.328394013997853e+00, -3.321704998072557e+00, -3.321704998072557e+00, -5.285648176528915e-01, -5.280540697811195e-01, -5.157644759480561e-01, -5.262471594178942e-01, -5.283860062586383e-01, -5.283860062586383e-01, -1.864947345741487e-01, -1.880585035696930e-01, -6.031698442826808e-01, -9.785742075442895e-02, -1.867884493215671e-01, -1.867884493215671e-01, -6.816947233667174e-03, -7.228216573991957e-03, -3.422294150944239e-02, -2.537562847997556e-03, -7.151162175255481e-03, -7.151162175255481e-03, -5.665924208698071e+00, -5.666990599569478e+00, -5.666044868785538e+00, -5.666873491639689e+00, -5.666455341962406e+00, -5.666455341962406e+00, -2.080485681060384e+00, -2.102994875377000e+00, -2.078059800418224e+00, -2.095656991689131e+00, -2.099134299026587e+00, -2.099134299026587e+00, -6.057541409526390e-01, -6.626878967758856e-01, -5.492103705931906e-01, -5.795850219300508e-01, -6.383299565104349e-01, -6.383299565104349e-01, -7.090971688033174e-02, -1.825508615658922e-01, -6.995824353227148e-02, -1.924764831172268e+00, -8.465911850342119e-02, -8.465911850342119e-02, -2.429984163860574e-03, -2.907871274174492e-03, -2.160742414113088e-03, -4.599454355488614e-02, -2.653525287857764e-03, -2.653525287857764e-03, -6.509357566444145e-01, -6.474305447702068e-01, -6.486940580193662e-01, -6.496637513476522e-01, -6.491782793966733e-01, -6.491782793966733e-01, -6.283594142671086e-01, -5.441342697388236e-01, -5.687659511716422e-01, -5.915889827286953e-01, -5.798076783727271e-01, -5.798076783727271e-01, -6.832801962115530e-01, -2.422480570819514e-01, -2.835334405687968e-01, -3.503586920455978e-01, -3.194648841528774e-01, -3.194648841528774e-01, -4.755614276738567e-01, -3.181665552144851e-02, -4.222540251634953e-02, -3.321790940813541e-01, -5.799191327398338e-02, -5.799191327398341e-02, -8.170393678070985e-03, -5.142964283801894e-04, -1.239186323174895e-03, -5.525118773018146e-02, -2.203546802793267e-03, -2.203546802793263e-03]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_mgga_x_lak_BrOH_1_vrho():
    # Prepare the input
    inp = test_data["BrOH"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_lak", 1)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-3.048804670789735e+01, -3.048813729566089e+01, -3.048860022292199e+01, -3.048711779227738e+01, -3.048809396170338e+01, -3.048809396170338e+01, -4.949302882815692e+00, -4.949397409166899e+00, -4.952097749215048e+00, -4.947048088732381e+00, -4.949381110521661e+00, -4.949381110521661e+00, -7.339216901680914e-01, -7.323883421347299e-01, -7.078742403413071e-01, -7.334814245418636e-01, -7.333917327378933e-01, -7.333917327378933e-01, -1.760895753596024e-01, -1.823484037020177e-01, -7.812335215106443e-01, -1.420761864052517e-01, -1.775172328588303e-01, -1.775172328588303e-01, -1.146155545626779e-02, -1.209075206471317e-02, -4.327696376544726e-02, -4.556131469627762e-03, -1.196115050988203e-02, -1.196115050988203e-02, -7.472452396531810e+00, -7.474455336827287e+00, -7.472667030666170e+00, -7.474223916666284e+00, -7.473466498563195e+00, -7.473466498563195e+00, -2.546574488135855e+00, -2.596597775028911e+00, -2.544178153267614e+00, -2.585088529159730e+00, -2.584936595439764e+00, -2.584936595439764e+00, -8.220796638183567e-01, -9.328685840834995e-01, -7.878968563657790e-01, -8.674808889020279e-01, -8.569426633688327e-01, -8.569426633688327e-01, -8.710369561812432e-02, -8.304918428485437e-02, -8.681523787454733e-02, -2.924801134622067e+00, -8.515718612413173e-02, -8.515718612413173e-02, -4.368260634360851e-03, -5.185920300210776e-03, -3.859523560364737e-03, -5.281223719009966e-02, -4.731002802772937e-03, -4.731002802772937e-03, -8.690871015289758e-01, -8.615645156263897e-01, -8.642411602413382e-01, -8.663217387398934e-01, -8.652793526620767e-01, -8.652793526620767e-01, -8.427273930713345e-01, -7.023557118746402e-01, -7.428695275907593e-01, -7.816338340343481e-01, -7.618251841665329e-01, -7.618251841665329e-01, -9.906351701341831e-01, -2.392891691747387e-01, -3.015925110839901e-01, -4.562765897295604e-01, -3.762626822583632e-01, -3.762626822583631e-01, -6.055713480777515e-01, -4.188300784245864e-02, -4.998694061142135e-02, -4.540995219616671e-01, -6.738471919682094e-02, -6.738471919682107e-02, -1.360610186292772e-02, -9.656587584572296e-04, -2.274956282788353e-03, -6.495648263052682e-02, -3.945017858138946e-03, -3.945017858138937e-03]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_x_lak_BrOH_1_vsigma():
    # Prepare the input
    inp = test_data["BrOH"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_lak", 1)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [-4.034971029265319e-09, -4.034909896962239e-09, -4.034353496268917e-09, -4.035338372563081e-09, -4.034941441695149e-09, -4.034941441695149e-09, -2.021787011924327e-05, -2.021784417896360e-05, -2.021586392745742e-05, -2.023162974027330e-05, -2.021733931429903e-05, -2.021733931429903e-05, -4.804907967664571e-03, -4.773933393196974e-03, -3.954599563203836e-03, -5.239522251765114e-03, -4.795333708801327e-03, -4.795333708801327e-03, -2.110086976915669e+00, -2.081155690999147e+00, -9.466369055664953e-04, 3.845577899835285e-02, -2.117138520688945e+00, -2.117138520688945e+00, 1.054859110900134e+02, 9.523459327877116e+01, -1.013030494453793e+00, 3.972944802402188e+02, 9.891992895483948e+01, 9.891992895483953e+01, -1.017084676514784e-06, -1.016211010722435e-06, -1.016879305920942e-06, -1.016207046631811e-06, -1.016792679083224e-06, -1.016792679083224e-06, -1.026155948148215e-04, -8.958076316993471e-05, -9.771909946311306e-05, -8.722869082948278e-05, -9.955611246174743e-05, -9.955611246174743e-05, -2.109922510988807e-02, -1.360296116288281e-02, -3.210931173492516e-02, -2.444386195281899e-02, -1.480682771272131e-02, -1.480682771272131e-02, -4.876400069170888e-01, -2.254067275286419e+00, -5.537678833784787e-01, -1.971887824000210e-04, -2.939338831579642e+00, -2.939338831579642e+00, 4.385161898953671e+02, 3.412423679788752e+02, 1.025122281086411e+03, -1.337037812059124e+00, 4.997567711817563e+02, 4.997567711817559e+02, -6.827414209630283e-03, -6.885356794616095e-03, -6.862567082411050e-03, -6.847700575701843e-03, -6.855439523544123e-03, -6.855439523544120e-03, -9.027536982363433e-03, -1.407486085041712e-02, -1.277298021199698e-02, -1.120613417179380e-02, -1.221753756991827e-02, -1.221753756991829e-02, -1.295989655484448e-02, -5.462816116349450e-01, -3.904914193852390e-01, -1.909981741481713e-01, -2.625571966906247e-01, -2.625571966906251e-01, -5.202327644438036e-02, -4.798252670177729e-01, -1.281007980312189e+00, -2.457969639384004e-01, -1.342410257206788e+00, -1.342410257206778e+00, 6.527184337175638e+01, 4.658692752626992e+03, 1.403246615317673e+03, -1.405914078501883e+00, 8.303248379975182e+02, 8.303248379975197e+02]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_x_lak_BrOH_1_vlapl():
    # Prepare the input
    inp = test_data["BrOH"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_lak", 1)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vlapl"].flatten()
    ref_tgt = [0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_x_lak_BrOH_1_vtau():
    # Prepare the input
    inp = test_data["BrOH"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_lak", 1)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vtau"].flatten()
    ref_tgt = [6.703809628635495e-04, 6.703755211836640e-04, 6.703106717066964e-04, 6.703971679901515e-04, 6.703784761422764e-04, 6.703784761422764e-04, 1.342814402234524e-02, 1.342865453511826e-02, 1.344397322815644e-02, 1.344268784782309e-02, 1.342818561329574e-02, 1.342818561329574e-02, 1.593212175643141e-02, 1.579468847541578e-02, 1.350809995221182e-02, 1.964450033166089e-02, 1.588948407307671e-02, 1.588948407307671e-02, 2.095143071130800e-01, 2.146797151969752e-01, 1.131623342844250e-03, 5.389433944223569e-03, 2.126027980561003e-01, 2.126027980561003e-01, 1.134694436962888e-05, 7.587700212864780e-06, 1.349953834746041e-04, 2.878293613594174e-12, 1.197391626661542e-05, 1.197391626661464e-05, 2.433978148647156e-03, 2.432476360282098e-03, 2.433561972791271e-03, 2.432410397163222e-03, 2.433562223435088e-03, 2.433562223435088e-03, 1.269626688530895e-02, 1.139369094494426e-02, 1.203872586490838e-02, 1.098845570367002e-02, 1.261790677099475e-02, 1.261790677099475e-02, 6.720826523337731e-02, 5.667935411051516e-02, 8.627847213565781e-02, 7.951553225685706e-02, 5.348367278051484e-02, 5.348367278051484e-02, 4.677016385016313e-03, 2.274974022023197e-01, 5.780222538766220e-03, 2.372318937750805e-02, 6.922717528443960e-02, 6.922717528443960e-02, 2.628554908303983e-11, 1.646092828645361e-11, 2.839259422808171e-10, 6.285570464492666e-05, 6.980097097686138e-12, 6.980097097686099e-12, 2.348899078087482e-02, 2.380156098291863e-02, 2.368528353600468e-02, 2.359828034791733e-02, 2.364311715841891e-02, 2.364311715841889e-02, 2.819037530205835e-02, 3.064957656828716e-02, 3.136580231176957e-02, 3.074786421082966e-02, 3.167718724952029e-02, 3.167718724952034e-02, 6.284221528602239e-02, 1.118167145119067e-01, 1.320326772788162e-01, 1.295082931417199e-01, 1.267260599224442e-01, 1.267260599224444e-01, 7.801801689424437e-02, 3.556749233700998e-04, 2.471369770232007e-04, 1.407286825839976e-01, 6.213299771098910e-03, 6.213299771098849e-03, 2.003771143618241e-09, 6.898864089465778e-16, 1.642378897074276e-11, 5.083422245998940e-03, 3.979110594156728e-12, 3.979110594156715e-12]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05