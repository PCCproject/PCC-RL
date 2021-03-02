package unocard;

import game.SpecialEffectsManager;

public class ReverseCard extends ActionCard{
	
	public ReverseCard(Color color) {
		super(color, ActionCard.Content.REVERSE);
	}
	
	/**
	 * Order of play switches directions (clockwise to counterclockwise, or vice versa)
	 * @param effManager: the object that controls the game effects.
	 */
	@Override
	public void doSpecialEffect(SpecialEffectsManager effManager){
		effManager.doReverse();
	}
}
